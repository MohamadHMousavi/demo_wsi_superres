import sys
import argparse
import torch
import data_loader as data
from trainer import Trainer


def parse_args(args):
    """
    parse args

    Parameters
    ----------
    args :
        passed arguments
    """
    parser = argparse.ArgumentParser(description='Train WSISR on compressed TMA dataset')
    parser.add_argument('--batch-size', default=16, type=int, help='Batch size')
    parser.add_argument('--patch-size', default=224, type=int, help='Patch size')
    parser.add_argument('--up-scale', default=2, type=float, help='Targeted upscale factor')
    parser.add_argument('--num-workers', default=4, type=int, help='Number of workers')
    parser.add_argument('--num-epochs', default=900, type=int,
                        help='Number of epochs, more epochs are desired for GAN training')
    parser.add_argument('--g-lr', default=0.0001, type=float, help='Learning rate of the generator')
    parser.add_argument('--d-lr', default=0.00001, type=float, help='Learning rate of the discriminator')
    parser.add_argument('--percep-weight', default=0.01, type=float, help='GAN loss weight')
    parser.add_argument('--run-from', default=None, type=int,
                        help='Load weights from a previous run, use folder name in [weights] folder')
    parser.add_argument('--gan', default=1, type=int, help='Use GAN')
    parser.add_argument('--dis-freq', default=1, type=int, help='Interval of training the discriminator')

    # GPU settings
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu ids: e.g. 0, 1. -1 is no GPU')

    parser.add_argument('--test', help='test model.', action='store_true')

    parser.add_argument('--dir', default='results', type=str,
                        help='directory')

    return parser.parse_args(args)


def main(args):
    # data.generate_compress_csv()

    if not args.test:
        trainer = Trainer(args)
        trainer.train()
    else:
        raise NotImplementedError('tester not implemented yet')


if __name__ == '__main__':
    args = sys.argv[1:]
    args = parse_args(args)
    args.device = torch.device("cuda:%d" % args.gpu_id if torch.cuda.is_available() else "cpu")

    print(args)

    main(args)
