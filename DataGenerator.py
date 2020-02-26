import tensorflow as tf # tf.__version__=='1.14.0' and tf.keras.__version__=='2.2.4-tf'
import numpy as np
import os, multiprocessing, threading

from utils import _list_valid_filenames_in_directory

class DirectoryIterator (tf.keras.utils.Sequence):
    """Iterator capable of reading datas from a directory on disk.

    # Arguments
        directory: string, path to the directory to read datas from.
            Each subdirectory in this directory will be
            considered to contain datas from one class,
            or alternatively you could specify class subdirectories
            via the `classes` argument.
        data_generator: Instance of `DataGenerator`
            to use for Data Extraction and preprocessing .
        path_filter: is a callable filter 
            which get path and return True if path is valid else False
        classes: Optional list of strings, names of subdirectories
            containing datas from each class.
            It will be computed automatically if not set.
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `"input"`: targets are datas identical to input datas (mainly
                used to work with autoencoders),
            `None`: no targets get yielded (only input datas are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
            If set to False, sorts the data in alphanumeric order.
        seed: Random seed for data shuffling.
        follow_links: boolean,follow symbolic links to subdirectories
        subset: Subset of data (`"training"` or `"validation"`) if
            validation_split is set in DataGenerator.
    """
    def __init__(
            self,
            directory,
            data_generator,
            path_filter= lambda x:True,
            classes= None,
            class_mode= 'categorical',
            batch_size= 32,
            shuffle= True,
            seed= None,
            follow_links= False,
            subset=None ):

        allowed_class_modes = {'binary', 'input', None, 'categorical', 'sparse'}

        self.data_generator = data_generator

        self.directory = directory
        self.path_filter = path_filter

        if subset is not None:
            validation_split = self.data_generator._validation_split
            if subset == 'validation':
                split = (0, validation_split)
            elif subset == 'training':
                split = (validation_split, 1)
            else:
                raise ValueError(
                    'Invalid subset name: %s;'
                    'expected "training" or "validation"' % (subset,))
        else:
            split = None
        self.split = split
        self.subset = subset

        self.classes = classes
        if class_mode not in allowed_class_modes:
            raise ValueError('Invalid class_mode: {}; expected one of: {}'
                             .format(class_mode, allowed_class_modes))
        self.class_mode = class_mode
        self.dtype = self.data_generator.dtype
        self.shape = self.data_generator.reshape
        # First, count the number of samples and classes.
        self.samples = 0

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)
        self.num_classes = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        pool = multiprocessing.pool.ThreadPool()

        # Second, build an index of the datas
        # in the different class subfolders.
        results = []
        self.filenames = []
        i = 0
        for dirpath in (os.path.join(directory, subdir) for subdir in classes):
            results.append(
                pool.apply_async(_list_valid_filenames_in_directory,
                                 (dirpath, self.data_generator.white_list_formats, self.split, self.path_filter,
                                  self.class_indices, follow_links)))
        classes_list = []
        for res in results:
            classes, filenames = res.get()
            classes_list.append(classes)
            self.filenames += filenames
        self.samples = len(self.filenames)
        self.classes = np.zeros((self.samples,), dtype='int32')
        for classes in classes_list:
            self.classes[i:i + len(classes)] = classes
            i += len(classes)

        print('Found %d files belonging to %d classes.' %
              (self.samples, self.num_classes))
        pool.close()
        pool.join()
        self._filepaths = [
            os.path.join(self.directory, fname) for fname in self.filenames
        ]

        self.n = self.samples
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_array = None
        self.index_generator = self._flow_index()

    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            self.index_array = np.random.permutation(self.n)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, '
                             'but the Sequence '
                             'has length {length}'.format(idx=idx,
                                                          length=len(self)))
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()
        index_array = self.index_array[self.batch_size * idx:
                                       self.batch_size * (idx + 1)]
        return self._get_batches_of_transformed_samples(index_array)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size  # round up

    def on_epoch_end(self):
        self._set_index_array()

    def reset(self):
        self.batch_index = 0

    def _flow_index(self):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            if self.batch_index == 0:
                self._set_index_array()

            if self.n == 0:
                # Avoiding modulo by zero error
                current_index = 0
            else:
                current_index = (self.batch_index * self.batch_size) % self.n
            if self.n > current_index + self.batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0
            self.total_batches_seen += 1
            yield self.index_array[current_index:
                                   current_index + self.batch_size]

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The preprocessing of datas is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)

    def _get_batches_of_transformed_samples(self, index_array): ##### p2d, d2D
        """Gets a batch of transformed samples.

        # Arguments
            index_array: Array of sample indices to include in batch.

        # Returns
            A batch of transformed samples.
        """
        batch_x = []#np.zeros((len(index_array),) + self.data_generator.reshape, dtype=self.data_generator.dtype)
        # build batch of data
        # self.filepaths is dynamic, is better to call it once outside the loop
        filepaths = self.filepaths
        for i, j in enumerate(index_array):
            x = self.data_generator.p2d(filepaths[j], self)
            x = self.data_generator.d2D(x, self)
            x = x.astype(self.dtype)
            x = x.reshape(self.shape)
            batch_x.append(x)
        batch_x = np.array(batch_x)

        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode in {'binary', 'sparse'}:
            batch_y = np.empty(len(batch_x), dtype=self.dtype)
            for i, n_observation in enumerate(index_array):
                batch_y[i] = self.classes[n_observation]
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), len(self.class_indices)),
                               dtype=self.dtype)
            for i, n_observation in enumerate(index_array):
                batch_y[i, self.classes[n_observation]] = 1.
        elif self.class_mode == 'multi_output':
            batch_y = [output[index_array] for output in self.labels]
        elif self.class_mode == 'raw':
            batch_y = self.labels[index_array]
        else:
            return batch_x
        if self.sample_weight is None:
            return batch_x, batch_y
        else:
            return batch_x, batch_y, self.sample_weight[index_array]

    @property
    def filepaths(self):
        return self._filepaths

    @property
    def labels(self):
        return self.classes

    @property  # mixin needs this property to work
    def sample_weight(self):
        # no sample weights will be returned
        return None

    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            self.index_array = np.random.permutation(self.n)

class DataGenerator ():
    """Generate batches of tensor data with real-time data augmentation.
     The data will be looped over (in batches).

    # Arguments
        DataExtracting_function: function, which gets path and instance of DataGenerator
            and return data (numpy.array) from the file in the given path
        reshape: tuple, shape by which data is reshaped for feeding into the model
        preprocessing_function : function, which get the data and return preprocessed data (numpy.array)
        white_list_formats: list, of extensitions.
            only filenames with these extensions are accepted for DataExtraction
        dtype: Dtype to use for the generated arrays.
        validation_split: Float. Fraction of datas reserved for validation
            (strictly between 0 and 1).
        kwargs: other attributes which is added to instance of DataGenerator,
            can be used by DataExtracting_function and preprocessing_function
    """
    def __init__(
            self,
            DataExtracting_function,
            reshape,
            preprocessing_function= lambda *x:x[0],
            white_list_formats= all,
            dtype= None,
            validation_split= None,
            **kwargs) :

        self.p2d = DataExtracting_function
        self.d2D = preprocessing_function
        self.white_list_formats = white_list_formats
        self.reshape = reshape
        self.dtype = dtype
        self.validation_split = validation_split

        for key, value in kwargs.items() :
            setattr(self, key, value)

    def flow_from_directory(
            self,
            directory,
            path_filter= lambda x:True,
            classes= None,
            class_mode= 'categorical',
            batch_size= 32,
            shuffle= True,
            seed= None,
            follow_links= False,
            subset= None ):
        """Takes the path to a directory & generates batches of augmented data.
    # Arguments
        directory: string, path to the directory to read datas from.
            Each subdirectory in this directory will be
            considered to contain datas from one class,
            or alternatively you could specify class subdirectories
            via the `classes` argument.
        data_generator: Instance of `DataGenerator`
            to use for Data Extraction and preprocessing .
        path_filter: is a callable filter 
            which get path and return True if path is valid else False
        classes: Optional list of strings, names of subdirectories
            containing datas from each class.
            It will be computed automatically if not set.
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `"input"`: targets are datas identical to input datas (mainly
                used to work with autoencoders),
            `None`: no targets get yielded (only input data are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
            If set to False, sorts the data in alphanumeric order.
        seed: Random seed for data shuffling.
        follow_links: boolean,follow symbolic links to subdirectories
        subset: Subset of data (`"training"` or `"validation"`) if
            validation_split is set in DataGenerator.

        # Returns
            A `DirectoryIterator` yielding tuples of `(x, y)`
                where `x` is a numpy array containing a batch
                of datas with shape `(batch_size, *shape)`
                and `y` is a numpy array of corresponding labels.
        """
        return DirectoryIterator(
                    directory,
                    self,
                    path_filter,
                    classes,
                    class_mode,
                    batch_size,
                    shuffle,
                    seed,
                    follow_links,
                    subset )

"""ToCheck :)
import librosa #For_example: isolated-word recognition
def P2D_mfcc(path, dg):
    wave, sr = librosa.load(path, mono=True)
    mfccs = librosa.feature.mfcc(y=wave, sr=sr, n_mfcc=dg.shape[0])
    mfccs = np.pad(mfccs, ((0,0), (0, dg.shape[1]-len(mfccs[0]))), mode='constant')
    return mfccs
data_generator = DataGenerator(P2D_mfcc, (40,100,1)).flow_from_directory("path_of_dataset")
"""
