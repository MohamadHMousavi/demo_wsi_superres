from __future__ import print_function, division
import os, glob, random
import torch
import pandas as pd
from skimage import io, transform, img_as_float, color, img_as_ubyte, exposure
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as F
from PIL import Image, ImageFilter
from utils import find

plt.ion()  # interactive mode


class Compress_Dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.files_list = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        img_name = self.files_list.iloc[idx, 0]  # image path
        img = Image.open(img_name)
        if self.transform:
            sample = self.transform(img)
        return sample


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Rescale(object):
    def __init__(self, output_size, up_factor=5, stc=False):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.up_factor = up_factor
        self.stc = stc

    def __call__(self, img):
        img_high = img
        if self.stc == True:
            factor = max(1, np.random.normal(self.up_factor, 0.5))
        else:
            factor = self.up_factor
        img_low = img_high.resize((int(img.size[1] / factor), int(img.size[0] / factor)))
        img_low = img_low.resize(self.output_size, Image.BILINEAR)
        img_low = img_low.filter(ImageFilter.GaussianBlur(radius=((factor - 1) / 2)))

        return {'input': img_low, 'output': img_high}


class ToHSV(object):
    def __call__(self, sample):
        img_low, img_high = sample['input'], sample['output']
        img_low = img_low.convert('HSV')
        img_high = img_high.convert('HSV')
        return {'input': img_low, 'output': img_high}


class ToTensor(object):
    def __call__(self, sample):
        img_low, img_high = sample['input'], sample['output']
        return {'input': transforms.functional.to_tensor(img_low), 'output': transforms.functional.to_tensor(img_high)}


def show_patch(dataloader, index=0, is_hsv=False):
    for i_batch, sample_batched in enumerate(dataloader):
        if i_batch == index:
            input_batch, output_batch = sample_batched['input'], sample_batched['output']
            if is_hsv:
                input_img = input_batch.numpy().transpose((0, 2, 3, 1))
                output_img = output_batch.numpy().transpose((0, 2, 3, 1))
                for i in range(0, input_batch.shape[0]):
                    input_img[i] = color.hsv2rgb(input_img[i])
                    output_img[i] = color.hsv2rgb(output_img[i])
                input_batch = torch.from_numpy(input_img.transpose(((0, 3, 1, 2))))
                output_batch = torch.from_numpy(output_img.transpose(((0, 3, 1, 2))))
            batch_size = len(input_batch)
            im_size = input_batch.size(2)
            plt.figure(figsize=(20, 10))
            grid = utils.make_grid(input_batch)
            plt.imshow(grid.numpy().transpose((1, 2, 0)), interpolation='bicubic')
            plt.axis('off')
            plt.figure(figsize=(20, 10))
            grid = utils.make_grid(output_batch)
            plt.imshow(grid.numpy().transpose((1, 2, 0)), interpolation='bicubic')
            plt.axis('off')
            break


# def generate_compress_csv():
#     train_imgs = glob.glob('dataset/TMA/20x-cores-training/*.jpg')
#     random.shuffle(train_imgs)
#     train_df = pd.DataFrame(train_imgs[0:int(0.8 * len(train_imgs))])
#     valid_df = pd.DataFrame(train_imgs[int(0.8 * len(train_imgs)):int(0.9 * len(train_imgs))])
#     test_df = pd.DataFrame(train_imgs[int(0.9 * len(train_imgs)):])
#     train_df.to_csv('dataset/TMA/train-compress.csv', index=False)
#     valid_df.to_csv('dataset/TMA/valid-compress.csv', index=False)
#     test_df.to_csv('dataset/TMA/test-compress.csv', index=False)


# def compress_csv_path(csv='train'):
#     if csv == 'train':
#         return 'dataset/TMA/train-compress.csv'
#     if csv == 'test':
#         return 'dataset/TMA/test-compress.csv'
#     if csv == 'valid':
#         return 'dataset/TMA/valid-compress.csv'

def compress_csv_path(csv='train'):
    if csv == 'train':
        return 'dataset/train-compress.csv'
    if csv == 'test':
        return 'dataset/test-compress.csv'
    if csv == 'valid':
        return 'dataset/valid-compress.csv'


def write_split_csv(save_dir, csv_dir, train_percent=0.8, test_percent=0.1):
    os.makedirs(save_dir, exist_ok=True)

    slides = find('*.csv', csv_dir)
    random.shuffle(slides)

    test_num = min(1, int(test_percent * len(slides)))

    test_slides = slides[:test_num]
    train_valid_slides = slides[test_num:]

    test_set = pd.DataFrame()
    for s in test_slides:
        test_set = test_set.append(pd.read_csv(s))

    train_valid_set = pd.DataFrame()
    for s in train_valid_slides:
        train_valid_set = train_valid_set.append(pd.read_csv(s))

    train_valid_shuffled = train_valid_set.sample(frac=1)

    # Calculate index for split 80:20 ratio
    train_valid_index = round(len(train_valid_shuffled) * train_percent)

    # Split into training and test sets
    train_set = train_valid_shuffled[:train_valid_index].reset_index(drop=True)
    valid_set = train_valid_shuffled[train_valid_index:].reset_index(drop=True)

    train_set.to_csv(os.path.join(save_dir, 'train-compress.csv'), index=False)
    test_set.to_csv(os.path.join(save_dir, 'test-compress.csv'), index=False)
    valid_set.to_csv(os.path.join(save_dir, 'valid-compress.csv'), index=False)


def move_dataset(csv='train'):
    from shutil import copyfile

    save_dir = '/media/jahanifar/Data/SuperRes'
    compress_csv_path(csv)
    train_paths = pd.read_csv(compress_csv_path(csv))
    if not os.path.exists(save_dir):
        raise NotADirectoryError(save_dir + ' not found')
    os.makedirs(os.path.join(save_dir, csv))
    new_train_paths = []
    for p in train_paths['path']:
        p_split = p.split('/')
        save_path = os.path.join(os.path.join(save_dir, csv), p_split[-3] + '_' + p_split[-2] + '_' + p_split[-1])
        new_train_paths.append(save_path)
        copyfile(p, save_path)

    df = pd.DataFrame({'path': new_train_paths})
    df.to_csv('dataset/' + csv + '-compress_ssd.csv', index=False)


if __name__ == '__main__':
    # write_split_csv('dataset', '/home/jahanifar/PycharmProjects/DataReader/image_path_csv')
    move_dataset(csv='train')
    print('end')