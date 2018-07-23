import scipy
import scipy.misc
import numpy as np


def get_image(image_path, resize_w, resize_h, is_resize=True):
    return transform(imread(image_path), resize_w, resize_h, is_resize)

def imread(path):
    return scipy.misc.imread(path, mode='RGB').astype(np.float)

def transform(image, resize_w, resize_h, is_resize=False):
    if is_resize:
        cropped_image = resize(image, resize_w, resize_h)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def resize(x, resize_w, resize_h):
    h, w = x.shape[:2]
    return scipy.misc.imresize(x[0:h, 0:w], [resize_w, resize_h])

def inverse_transform(images):
    return (images+1.)/2.

def save_images(images, size, i_w, i_h, image_path):
    images = images.reshape([-1,i_w,i_h,3])
    return imsave(inverse_transform(images), size, image_path)

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((int(h * size[0]), int(w * size[1]), 3))
    for idx, image in enumerate(images):
        i = int(idx % size[1])
        j = int(idx // size[1])
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img