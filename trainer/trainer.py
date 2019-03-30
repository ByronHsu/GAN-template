import torch
import logging
import os
import shutil
from torchvision.utils import save_image

class Trainer:    
    def __init__(self, generator, discriminator, dataloader, opt):
        self.dataloader = dataloader
        self.n_epochs = opt.n_epochs
        self.device = self._prapare_gpu()
        self.G = generator.to(self.device)
        self.D = discriminator.to(self.device)
        self.opt = opt
        self.logger = logging.getLogger(self.__class__.__name__)
        self.begin_epoch = 0
        self._resume_checkpoint(opt.resume)

    def _prapare_gpu(self):
        n_gpu = torch.cuda.device_count()
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        return device

    def _resume_checkpoint(self, path):
        if path == None: return
        try:
            checkpoint = torch.load(path)
            self.G.load_state_dict(checkpoint['G'])
            self.D.load_state_dict(checkpoint['D'])
            self.begin_epoch = checkpoint['log']['epoch']
        except:
            self.logger.error('[Resume] Cannot load from checkpoint')

    def train(self):
        ensure_dir(self.opt.check_dir)
        ensure_dir(self.opt.image_dir)
        logger = self.logger
        logger.info('[Model Structure]')
        logger.info(self.G)
        logger.info(self.D)

        for epoch in range(self.begin_epoch, self.begin_epoch + self.n_epochs):
            log, gen_imgs = self._train_epoch(epoch)
            
            logger.info('=========================='.format(epoch))

            checkpoint = {
                'log': log,
                'G': self.G.state_dict(),
                'D': self.D.state_dict(),
                'gen_imgs': gen_imgs,
            }

            logger.info('> Epoch: {}\n> g_loss: {}\n> d_loss: {}'.format(log['epoch'], log['g_loss'], log['d_loss']))

            # save checkpoint
            check_path = os.path.join(self.opt.check_dir, 'checkpoint.pth')
            torch.save(checkpoint, check_path)
            logger.info('> checkpoint: {}'.format(check_path))
            # save generated images
            if epoch % self.opt.sample_interval == 0:
                image_path = os.path.join(self.opt.image_dir, 'epoch{}.png'.format(epoch))
                save_image(gen_imgs, image_path, nrow=5, normalize=True)
                logger.info('> image: {}'.format(image_path))

            logger.info('=========================='.format(epoch))

    def _train_epoch(self, epoch):
        # Reassign class member
        opt, dataloader, G, D, device, logger = self.opt, self.dataloader, self.G, self.D, self.device, self.logger
        # Optimizers
        optimizer_G = torch.optim.Adam(G.parameters(), lr = opt.lr, betas=(opt.b1, opt.b2))
        optimizer_D = torch.optim.Adam(D.parameters(), lr = opt.lr, betas=(opt.b1, opt.b2))
        # Loss function
        adversarial_loss = torch.nn.BCELoss()
        # Record some variable
        total_G_loss = 0
        total_D_loss = 0
        total_gen_imgs = torch.Tensor().to(device)

        for i, imgs in enumerate(dataloader):
            # Adversarial ground truths
            valid = torch.Tensor(imgs.size(0), 1).fill_(1.0).to(device)
            fake = torch.Tensor(imgs.size(0), 1).fill_(0.0).to(device)
            imgs = imgs.to(device)
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            # Sample noise as generator input
            z = torch.rand(imgs.shape[0], opt.latent_dim).to(device)
            # Generate a batch of images
            gen_imgs = G(z)
            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(D(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(D(imgs), valid)
            fake_loss = adversarial_loss(D(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            
            # Show and save the result
            total_G_loss += g_loss.item()
            total_D_loss += d_loss.item()
            
            total_gen_imgs = torch.cat((total_gen_imgs, gen_imgs), dim = 0)

            logger.info("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (
                epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item()
            ))

        log = {
            'epoch': epoch,
            'g_loss': total_G_loss / len(dataloader),
            'd_loss': total_D_loss / len(dataloader),
        }
        # random shuffle gen imgs and return the first 25 images
        idxs = torch.randperm(total_gen_imgs.shape[0])
        total_gen_imgs = total_gen_imgs[idxs]
        return log, total_gen_imgs[:25]

def ensure_dir(dir):
    os.makedirs(dir, exist_ok=True)