import os
import sys

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import models

import data_loader as data
from torch.utils.data import DataLoader
from torchvision import transforms

from math import log2
from tester import Tester


def get_dataloader(num_workers, batch_size, patch_size, cur_factor, csv='train', stc=False):
    transformed_dataset = data.Compress_Dataset(csv_file=data.compress_csv_path(csv),
                                                transform=data.Compose([
                                                    transforms.RandomCrop((patch_size, patch_size)),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.RandomVerticalFlip(),
                                                    data.Rescale((patch_size, patch_size),
                                                                 up_factor=cur_factor, stc=stc),
                                                    data.ToTensor()
                                                ]))
    dataloader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader


class Trainer:
    def __init__(self, args):
        self.gan_c = args.percep_weight
        self.gan = args.gan
        self.device = args.device
        self.patch = (1, args.patch_size // 2 ** 4, args.patch_size // 2 ** 4)
        self.dis_freq = args.dis_freq
        self.num_epochs = args.num_epochs

        self.patch_size = args.patch_size
        self.batch_size = args.batch_size
        self.dis_out_shape = (self.batch_size, 1, self.patch_size // 2 ** 4, self.patch_size // 2 ** 4)

        self.num_workers = args.num_workers
        self.up_scale = args.up_scale


        self.generator = models.Generator()
        self.generator.to(self.device)
        self.discriminator = models.Discriminator()
        self.discriminator.to(self.device)

        self.tester = Tester(args, 'output')

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=args.g_lr)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=args.d_lr)
        self.criterionL = nn.L1Loss().to(self.device)
        self.criterionMSE = nn.MSELoss().to(self.device)
        self.scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_G, self.num_epochs,
                                                                      args.g_lr * 0.05)
        self.scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_D, self.num_epochs,
                                                                      args.d_lr * 0.05)

        self.weight_dir = os.path.join(args.dir, 'weights')
        os.makedirs(self.weight_dir, exist_ok=True)
        self.log_path = os.path.join(args.dir, 'logs')
        os.makedirs(self.log_path, exist_ok=True)

        if args.run_from is not None:
            gen_path = os.path.join(self.weight_dir, 'generator_{}.pth'.format(args.run_from))
            if os.path.exists(gen_path):
                self.generator.load_state_dict(torch.load(gen_path))
            else:
                raise FileNotFoundError('Generator weights not found!')
            dis_path = os.path.join(self.weight_dir, 'discriminator_{}.pth'.format(args.run_from))
            if os.path.exists(dis_path):
                self.generator.load_state_dict(torch.load(dis_path))
            else:
                print('Discriminator weights not found!')
                pass
            self.start_epoch = args.run_from + 1
        else:
            self.start_epoch = 0

            # writing log for training
        self.writer = SummaryWriter(self.log_path)

    def epoch_train(self, dataloader, epoch):
        sum_gan_loss = 0
        sum_generator_loss = 0
        sum_discriminator_loss = 0
        epoch_loss = 0

        self.generator.train()
        for iteration, batch in enumerate(dataloader):
            real_mid = Variable(batch['input'].to(self.device), requires_grad=False)
            real_high = Variable(batch['output'].to(self.device), requires_grad=False)

            # Adversarial ground truths
            valid = Variable(torch.ones(self.dis_out_shape).to(self.device), requires_grad=False)
            fake = Variable(torch.zeros(self.dis_out_shape).to(self.device), requires_grad=False)

            # ---------------
            #  Train Generator
            # ---------------
            self.optimizer_G.zero_grad()
            # GAN loss
            fake_high = self.generator(real_mid)
            if self.gan:
                fake_prediction = self.discriminator(fake_high, real_mid)
                gan_loss = self.criterionMSE(fake_prediction, valid)
                sum_gan_loss += gan_loss.item()

            # Identity
            pixel_loss = self.criterionL(fake_high, real_high)
            epoch_loss += pixel_loss.item()

            # Total loss
            if self.gan:
                loss_G = self.gan_c * gan_loss + (1 - self.gan_c) * pixel_loss
                loss_G.backward()
                sum_generator_loss += loss_G.item()
            else:
                pixel_loss.backward()

            self.optimizer_G.step()

            # ---------------
            #  Train Discriminator
            # ---------------
            if self.gan and iteration % self.dis_freq == 0:
                self.optimizer_D.zero_grad()

                # Real loss
                pred_real = self.discriminator(real_high, real_mid)
                loss_real = self.criterionMSE(pred_real, valid)

                # Fake loss
                fake_prediction = self.discriminator(fake_high.detach(), real_mid)
                loss_fake = self.criterionMSE(fake_prediction, fake)

                # Total loss
                loss_D = 0.5 * (loss_real + loss_fake)
                loss_D.backward()

                self.optimizer_D.step()
                sum_discriminator_loss += loss_D.item()

            if self.gan:
                sys.stdout.write(
                    '\r[%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss (Identity/Advers/Total): %.4f/%.4f/%.4f'
                    % (epoch, self.num_epochs, iteration, len(dataloader), loss_D.item(),
                       pixel_loss.item(), gan_loss.item(), loss_G.item()))
            else:
                sys.stdout.write('\r[%d/%d][%d/%d] Generator_L1_Loss: %.4f'
                                 % (epoch, self.num_epochs, iteration, len(dataloader), pixel_loss.item()))

        if self.gan:
            self.writer.add_scalar("perceptron_loss", sum_gan_loss / len(dataloader), epoch + 1)
            self.writer.add_scalar("total_generator_loss", sum_generator_loss / len(dataloader), epoch + 1)
            self.writer.add_scalar("discriminator_loss", sum_discriminator_loss / len(dataloader), epoch + 1)
        self.writer.add_scalar("pixel_loss", epoch_loss / len(dataloader), epoch + 1)
        self.writer.flush()

        print("\n ===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(dataloader)))

        g_path = os.path.join(self.weight_dir, 'generator_{}.pth'.format(epoch))
        torch.save(self.generator.state_dict(), g_path)
        if self.gan:
            d_path = os.path.join(self.weight_dir, 'discriminator_{}.pth'.format(epoch))
            torch.save(self.discriminator.state_dict(), d_path)

    def train(self):
        cur_length = int(0.5 * self.num_epochs)
        init_scale = 2 ** 2
        step_size = (2 ** self.up_scale - init_scale) / cur_length
        for epoch in range(self.start_epoch, self.num_epochs):
            factor = min(log2(init_scale + (epoch - 1) * step_size), self.up_scale)
            print('curriculum updated: {} '.format(factor))
            train_dataset = get_dataloader(self.num_workers, self.batch_size, self.patch_size, factor, 'train',
                                           stc=True)

            self.epoch_train(train_dataset, epoch)
            self.scheduler_G.step()
            self.scheduler_D.step()

            if epoch % 1 == 0:
                fid, psnr = self.tester.test(self.generator, data.compress_csv_path('valid'))
                # print_output(generator, valid_dataset, device)
                print('\r>>>> PSNR: {}, FID: {}'.format(psnr, fid))

