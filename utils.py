import os
import warnings

import numpy as np

def _iter_valid_files(directory, white_list_formats, follow_links, path_filter):
    """Iterates on files with extension in `white_list_formats` contained in `directory`.

    # Arguments
        directory: Absolute path to the directory
            containing files to be counted
        white_list_formats: Set of strings containing allowed extensions for
            the files to be counted.
        follow_links: Boolean, follow symbolic links to subdirectories.
        path_filter: is a callable filter which get path and return True if path is valid else False

    # Yields
        Tuple of (root, filename) with extension in `white_list_formats`.
    """
    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=follow_links),
                      key=lambda x: x[0])

    for root, _, files in _recursive_list(directory):
        for fname in sorted(files):
            if (white_list_formats == all or fname.lower().endswith(white_list_formats)) and path_filter(os.path.join(directory,root,fname)):
                yield root, fname

def _list_valid_filenames_in_directory(directory, white_list_formats, split, path_filter,
                                       class_indices, follow_links):
    """Lists paths of files in `subdir` with extensions in `white_list_formats`.

    # Arguments
        directory: absolute path to a directory containing the files to list.
            The directory name is used as class label
            and must be a key of `class_indices`.
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.
        split: tuple of floats (e.g. `(0.2, 0.6)`) to only take into
            account a certain fraction of files in each directory.
            E.g.: `segment=(0.6, 1.0)` would only account for last 40 percent
            of images in each directory.
        path_filter: is a callable filter which get path and return True if path is valid else False
        class_indices: dictionary mapping a class name to its index.
        follow_links: boolean, follow symbolic links to subdirectories.

    # Returns
         classes: a list of class indices
         filenames: the path of valid files in `directory`, relative from
             `directory`'s parent (e.g., if `directory` is "dataset/class1",
            the filenames will be
            `["class1/file1.jpg", "class1/file2.jpg", ...]`).
    """
    dirname = os.path.basename(directory)
    if split:
        num_files = len(list(
            _iter_valid_files(directory, white_list_formats, follow_links)))
        start, stop = int(split[0] * num_files), int(split[1] * num_files)
        valid_files = list(
            _iter_valid_files(
                directory, white_list_formats, follow_links))[start: stop]
    else:
        valid_files = _iter_valid_files(
            directory, white_list_formats, follow_links, path_filter)
    classes = []
    filenames = []
    for root, fname in valid_files:
        classes.append(class_indices[dirname])
        absolute_path = os.path.join(root, fname)
        relative_path = os.path.join(
            dirname, os.path.relpath(absolute_path, directory))
        filenames.append(relative_path)

    return classes, filenames

