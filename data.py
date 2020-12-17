import keras
import numpy as np
import pandas as pd
from scipy import ndimage
from PIL import Image
import cv2
import PIL
from glob import glob
import warnings
import sys

warnings.filterwarnings("ignore")


def load_data(file_path, input_shape, normalize):
    if file_path.split('.')[-1] == 'npy':
        file_list, mask_list = np.load(file_path, allow_pickle=True)
        print('loading data from numpy file...')
    elif file_path.split('.')[-1] == 'csv':
        csv = pd.read_csv(file_path)
        file_list = csv['Image_name']
        mask_list = csv['Mask_name']
        print('loading data from csv file...')
    elif file_path[-1] == '/':
        file_list = glob(file_path + 'image/*')
        mask_list = glob(file_path + 'mask/*')
        print('loading data from directory...')
    else:
        print('Only support loading data from csv/npy file')
        sys.exit()

    x, y = [], []
    for idx in range(len(file_list)):
        img = keras.preprocessing.image.load_img(file_list[idx], target_size=input_shape, color_mode='grayscale')
        label = keras.preprocessing.image.load_img(mask_list[idx], target_size=input_shape, color_mode='grayscale')
        img = keras.preprocessing.image.img_to_array(img)
        label = keras.preprocessing.image.img_to_array(label)
        label[label < 0.5] = 0
        label[label > 0.5] = 1
        x.append(img)
        y.append(label)
    x, y = np.array(x), np.array(y)

    if normalize:
        print('data - normalized')
        x /= 255.
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        x[..., 0] -= mean[0]
        # x[..., 1] -= mean[1]
        # x[..., 2] -= mean[2]
        x[..., 0] /= std[0]
        # x[..., 1] /= std[1]
        # x[..., 2] /= std[2]
    else:
        print('data - scaled to [0, 1]')
        x = x / 255.
    return x, y, file_list


def img_multiply(x, y):
    factor = np.random.uniform(0.8, 1.2)
    return x * factor, y


def h_shift(x, y):
    h_index = np.random.randint(-20, 20)
    x_new = ndimage.shift(x, shift=[0, h_index, 0], mode='nearest')
    y_new = ndimage.shift(y, shift=[0, h_index, 0], mode='nearest')
    return x_new, y_new


def v_shift(x, y):
    v_index = np.random.randint(-20, 20)
    x_new = ndimage.shift(x, shift=[v_index, 0, 0], mode='nearest')
    y_new = ndimage.shift(y, shift=[v_index, 0, 0], mode='nearest')
    return x_new, y_new


def rotate(x, y):
    angle = np.random.choice((-20, 20))
    x_new = ndimage.rotate(x, angle, reshape=False)
    y_new = ndimage.rotate(y, angle, reshape=False)
    x_new = cv2.resize(x_new, (x.shape[0], x.shape[1]))
    y_new = cv2.resize(y_new, (y.shape[0], y.shape[1]))
    if len(x_new.shape) == 2:
        x_new = np.expand_dims(x_new, axis=-1)
        y_new = np.expand_dims(y_new, axis=-1)
    return x_new, y_new


def zoomin(x, y):
    factor = np.random.uniform(1.1, 1.3)
    x_new = cv2.resize(x, (int(x.shape[0] * factor), int(x.shape[1] * factor)))
    y_new = cv2.resize(y, (int(x.shape[0] * factor), int(x.shape[1] * factor)))
    length = int(x_new.shape[1] / 2)
    x_new = x_new[length - int(x.shape[1] / 2): length + int(x.shape[1] / 2), length - int(x.shape[1] / 2): length + int(x.shape[1] / 2), ]
    y_new = y_new[length - int(x.shape[1] / 2): length + int(x.shape[1] / 2), length - int(x.shape[1] / 2): length + int(x.shape[1] / 2), ]
    if len(x_new.shape) == 2:
        x_new = np.expand_dims(x_new, axis=-1)
        y_new = np.expand_dims(y_new, axis=-1)
    return x_new, y_new


def zoomout(x, y):
    factor = np.random.uniform(0.8, 0.9)
    x_new = cv2.resize(x, (int(x.shape[0] * factor), int(x.shape[1] * factor)))
    y_new = cv2.resize(y, (int(x.shape[0] * factor), int(x.shape[1] * factor)))
    if x.shape[-1] == 1:
        x_new = np.stack([x_new, x_new, x_new], axis=-1)
        y_new = np.stack([y_new, y_new, y_new], axis=-1)
    length = int(x_new.shape[0])
    a = int(0.5 * (x.shape[0] - length))
    b = int(0.5 * (x.shape[0] - length))
    if a + b + length == x.shape[0]:
        x_new = np.pad(x_new, ((a, b), (a, b), (0, 0)), mode='edge')
        y_new = np.pad(y_new, ((a, b), (a, b), (0, 0)), mode='edge')
    else:
        b = x.shape[0] - a - length
        x_new = np.pad(x_new, ((a, b), (a, b), (0, 0)), mode='edge')
        y_new = np.pad(y_new, ((a, b), (a, b), (0, 0)), mode='edge')
    if x.shape[-1] == 1:
        x_new = np.expand_dims(x_new[..., 0], axis=-1)
        y_new = np.expand_dims(y_new[..., 0], axis=-1)
    return x_new, y_new


def contrast(x, y):
    rand = np.random.uniform(1.2, 2)
    x = np.repeat(x, 3, axis=-1)
    min_val = np.min(x)
    diff_val = np.max(x) - np.min(x)
    x = (x - min_val) / diff_val * 255
    x_image = Image.fromarray(x.astype(np.uint8))
    contra = PIL.ImageEnhance.Contrast(x_image)
    x_new = contra.enhance(rand)
    x_new = np.reshape(x_new, (x.shape[0], x.shape[1], 3))
    x_new = (x_new[..., 0] + x_new[..., 1] + x_new[..., 2]) / 3
    return np.expand_dims(x_new / 255 * diff_val + min_val, axis=-1), y


def data_augment(img, label, times=5):
    assert len(img) == len(label), "Images and labels must be of same length/shape!"
    img_aug, label_aug = [], []
    img, label = np.repeat(img, times, axis=0), np.repeat(label, times, axis=0)
    augmentation_list = [img_multiply, h_shift, v_shift, rotate, zoomin, zoomout, contrast]
    for idx in range(len(img)):
        x, y = img[idx], label[idx]
        x_new, y_new = np.random.choice(augmentation_list)(x, y)
        y_new[y_new < 0.5] = 0
        y_new[y_new > 0.5] = 1
        img_aug.append(x_new)
        label_aug.append(y_new)
    return np.array(img_aug), np.array(label_aug)
