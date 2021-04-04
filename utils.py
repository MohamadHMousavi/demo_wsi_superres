import fnmatch
import os
import numpy as np

from skimage.util import view_as_windows


def find(pattern, path, deep=False):
    """
    find pattern in directory

    Parameters
    ----------
    pattern : str
        pattern to look for in directory
    path : str
        directory to look in
    deep : bool
        look in sub-directories if True

    Returns
    -------
    list
        list of file paths containing pattern

    Examples
    --------
    look for all jpeg images in folder and its sub-directories

    >>> image_paths = find('*.jpg', 'folder', deep=True)
    """
    result = []
    for root, dirs, files in os.walk(path):
        if deep:
            for sub_dir in dirs:
                find(pattern, os.path.join(root, sub_dir), deep)
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def as_windows(img, step, patch_size):
    pad_h = int((np.floor(img.shape[0] / step) * step + patch_size) - img.shape[0])
    pad_w = int((np.floor(img.shape[1] / step) * step + patch_size) - img.shape[1])
    img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    img_wd = view_as_windows(img_padded, (patch_size, patch_size, 3), step=step)
    return np.squeeze(img_wd)
