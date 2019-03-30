import logging
from loader import DataLoader
from trainer import Trainer
from models import DC_Generator, DC_Discriminator
from utils import Train_options
import torch
import argparse

if __name__ == '__main__':
    # load arguments
    opt = Train_options()
    # set logging
    logging.basicConfig(level=logging.INFO, format='')
    logger = logging.getLogger('train')
    # set models and loader
    L = DataLoader(data_dir = opt.data_dir, batch_size = opt.batch_size, shuffle = True, validation_split = 0.0, num_workers = 0, opt = opt, training = True)
    G = DC_Generator(opt)
    D = DC_Discriminator(opt)
    # start trainer
    trainer = Trainer(generator = G, discriminator = D, dataloader = L, opt = opt)
    trainer.train()