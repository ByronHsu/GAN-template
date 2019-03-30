import torch.nn as nn
import numpy as np
import math
from .base_model import BaseModel

# DC GAN
class DC_Generator(BaseModel):
    def __init__(self, opt):
        super(DC_Generator, self).__init__()
        self.opt = opt
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 64 * (opt.img_height // 4) * (opt.img_width // 4)))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 64, self.opt.img_height // 4, self.opt.img_width // 4)
        img = self.conv_blocks(out)
        return img


class DC_Discriminator(BaseModel):
    def __init__(self, opt):
        super(DC_Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = (math.ceil(opt.img_height / 2 ** 4), math.ceil(opt.img_width / 2 ** 4))
        self.adv_layer = nn.Sequential(nn.Linear(128 * np.prod(ds_size), 1), nn.Sigmoid())
    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


# Traditional GAN
class Generator(BaseModel):
    """
    arguments:
        img_shape: (h, w, c)
        latent_dim: the dimension of the initial noise
    """
    def __init__(self, img_shape, latent_dim):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.model = nn.Sequential(
            *self._block(latent_dim, 128, normalize=False),
            *self._block(128, 256),
            *self._block(256, 512),
            *self._block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
    def _block(self, in_feat, out_feat, normalize=True):
        layers = [nn.Linear(in_feat, out_feat)]
        if normalize:
            layers.append(nn.BatchNorm1d(out_feat, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(BaseModel):
    """
    arguments:
        img_shape: (h, w, c)
    """
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity