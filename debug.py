import logging
from loader import DataLoader
from trainer import Trainer
import models
import torch
import argparse
ogging.basicConfig(level=logging.INFO, format='')
logger = logging.getLogger()

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--check_dir', type=str, default='save', help='the directory to store checkpoints')
parser.add_argument('--image_dir', type=str, default='gen_images', help='the directory to store images')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_height', type=int, default=100, help='height of each image')
parser.add_argument('--img_width', type=int, default=130, help='width of each image')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=400, help='interval betwen image samples')
parser.add_argument('--checkpoint', type=str, default=None, help='resume from a given checkpoint')

opt = parser.parse_args()

IMG_SHAPE = (3, 100, 130)
LATENT_DIM = 100
if __name__ == '__main__':

    X = DataLoader(data_dir = 'images', batch_size = opt.batch_size, shuffle = True, validation_split = 0.0, num_workers = 0, training = True)
    G = models.Generator(IMG_SHAPE, LATENT_DIM)
    D = models.Discriminator(IMG_SHAPE)
    trainer = Trainer(generator = G, discriminator = D, dataloader = X, opt = opt)
    
    trainer.train()