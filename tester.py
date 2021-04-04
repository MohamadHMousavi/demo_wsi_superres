import os, shutil
import pandas as pd
from PIL import Image, ImageFilter
from skimage import io, img_as_float, img_as_ubyte
import numpy as np

import torch
import torch.nn as nn

from utils import find, as_windows
import pytorch_fid.fid_score as fid_score

from math import log10


def p_snr(path_input, path_ref):
    MSE = nn.MSELoss()
    imgs_input = find('*.png', path_input)
    imgs_ref = find('*.png', path_ref)
    ave_psnr = 0
    for i in range(len(imgs_input)):
        img_input = torch.from_numpy(img_as_float(io.imread(imgs_input[i]).transpose(2, 1, 0)))
        img_ref = torch.from_numpy(img_as_float(io.imread(imgs_ref[i]).transpose(2, 1, 0)))
        img_input = img_input[None, :]
        img_ref = img_ref[None, :]
        mse = MSE(img_input, img_ref)
        psnr = 10 * log10(1 / mse.item())
        ave_psnr += psnr
    ave_psnr = ave_psnr / len(imgs_input)
    return ave_psnr


class Tester:
    def __init__(self, args, result_dir, overlap_percent=0.1):
        self.result_dir = result_dir
        shutil.rmtree(self.result_dir)
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(os.path.join(self.result_dir, 'temp_patch'), exist_ok=True)
        os.makedirs(os.path.join(self.result_dir, 'temp_patch_target'), exist_ok=True)

        self.up_scale = args.up_scale
        self.patch_size = args.patch_size
        self.step = int((1 - overlap_percent) * self.patch_size)

        self.device = args.device

    def test(self, generator, test_csv):
        test_files = pd.read_csv(test_csv)
        avg_fid = 0
        avg_psnr = 0
        for k in range(len(test_files)):
            img = Image.open(test_files.iloc[k, 0])
            img_hr_array = img_as_float(np.array(img))
            img_hr_wd = as_windows(img_hr_array, self.step, self.patch_size)

            img_lr = img.resize((int(img.size[1] / self.up_scale), int(img.size[0] / self.up_scale)))
            img_lr = img_lr.resize(img.size, Image.BILINEAR)
            img_lr = img_lr.filter(ImageFilter.GaussianBlur(radius=((self.up_scale - 1) / 2)))
            img_lr_array = img_as_float(np.array(img_lr))
            img_lr_wd = as_windows(img_lr_array, self.step, self.patch_size)

            with open(os.path.join(self.result_dir, 'temp_patch/TileConfiguration.txt'), 'w') as text_file:
                print('dim = {}'.format(2), file=text_file)
                with torch.no_grad():
                    generator.eval()
                    for i in range(0, img_lr_wd.shape[1]):
                        for j in range(0, img_lr_wd.shape[0]):
                            target = img_hr_wd[j, i]
                            patch = img_lr_wd[j, i].transpose((2, 0, 1))[None, :]
                            patch_tensor = torch.from_numpy(patch).float().to(self.device)
                            prediction = generator(patch_tensor)
                            io.imsave('output/temp_patch/{}_{}.png'.format(j, i),
                                      img_as_ubyte(np.clip(prediction.cpu().numpy()[0], 0, 1)))
                            io.imsave('output/temp_patch_target/{}_{}.png'.format(j, i), img_as_ubyte(target))
                            print('{}_{}.png; ; ({}, {})'.format(j, i, i * self.step, j * self.step), file=text_file)
            fid = fid_score.calculate_fid_given_paths((os.path.join(self.result_dir, 'output/temp_patch'),
                                                       os.path.join(self.result_dir, 'output/temp_patch_target')),
                                                      8, self.device, 2048)
            avg_fid = avg_fid + fid

            psnr = p_snr('output/temp_patch', 'output/temp_patch_target')
            avg_psnr = avg_psnr + psnr
            psnr = avg_psnr / len(test_files)

        fid = avg_fid / len(test_files)
        return fid, psnr
