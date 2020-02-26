# DataGenerator
Data Generator for training TensorFlow|Keras models 
To generate batches of data to train TensorFlow|Keras models

This is similar to `tf.keras.preprocessing.image.ImageDataGenerator`
where we have to give Data Extractor (which get path and return numpy data) and
Preprocessing_function (which get raw numpy data and return preprocessed numpy data).

This is a multipurpose data generator (eg: speech recognition which needs to convert .wav file to numpy array)
with similar approach to [tf.keras.preprocessing.image.ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator "Documentation"). so it is easy-to-use and generalized.

### Difference
* DataExtraction_function : which returns numpy array (data) from the given path
```python3
def p2d(path, DataIterator):
   """
   $path : path of file to be converted to numpy array (data)
   $DataIterator : instance of DataGenerator.flow_from_directory
     from where we get some details like required shape of data
   $return : numpy array (data) extracted from the file (with given path)
   """
   with open(path,'rb') as file :
       data = Extract(file.read())
   # do other stuff
   return data
```
* preprocessing_function  : which returns preprocessed numpy array from the given raw numpy array
```python3
def d2D(data, DataIterator):
    """
    $data : extracted non-preprocessed data (numpy array)
    $DataIterator : instance of DataGenerator.flow_from_directory
     from where we get some details like required shape of data
    $return : preprocessed numpy array
    """
    data = Preprocess(data)
    # do other stuff
    return data
```

#### Example
```python3
dg = DataGenerator(
         p2d, #DataExtracting_function
         (40,40,1), #Input_shape for first layer
         d2D #preprecessing_function
         )
DG = dg.flow_from_directory(
         "path_of_dataset"
         )
model.fit_generator(DG)
```
