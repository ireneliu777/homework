from typing import Union, Dict, Any, Tuple, Optional

import wandb
import torch
import torch.nn as nn
from torch import Tensor
from pytorch_lightning import LightningModule
import torch.nn as nn
import torch.autograd as autograd

from torch.utils.data import DataLoader

class MNISTGANModel(LightningModule):
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.generator = generator
        self.discriminator = discriminator
        self.adversarial_loss = torch.nn.MSELoss()
        self.opt_g = None
        self.opt_d = None

    def forward(self, z, labels) -> Tensor:
        return self.generator(z, labels)

    def configure_optimizers(self):
        self.opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.b1, self.hparams.b2),
        )
        self.opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.b1, self.hparams.b2)
        )
        return [self.opt_g, self.opt_d], []

    def training_step(self, batch, batch_idx, optimizer_idx) -> Union[Tensor, Dict[str, Any]]:
        log_dict, loss = self.step(batch, batch_idx, optimizer_idx)
        self.log_dict({"/".join(("train", k)): v for k, v in log_dict.items()})
        return loss

    def validation_step(self, batch, batch_idx) -> Union[Tensor, Dict[str, Any], None]:
        log_dict, loss = self.step(batch, batch_idx)
        self.log_dict({"/".join(("val", k)): v for k, v in log_dict.items()})
        return None

    def test_step(self, batch, batch_idx) -> Union[Tensor, Dict[str, Any], None]:
        # TODO: if you have time, try implementing a test step
        import torchvision.utils as vutils
        imgs, labels = batch
        generated_imgs = self.generator(z, labels)
        vutils.save_image(gen_imgs, "generated_images.png", normalize=True)
        loss = self.adversarial_loss(generated_imgs, imgs)
        return loss




    def step(self, batch, batch_idx, optimizer_idx=None) -> Tuple[Dict[str, Tensor], Optional[Tensor]]:
        if self.opt_g is None or self.opt_d is None:
            self.configure_optimizers()

        imgs, labels = batch
        batch_size = imgs.shape[0]

        log_dict = {}
        loss = None

        # Adversarial ground truths
        valid = torch.ones(batch_size, 1)
        fake = torch.zeros(batch_size, 1)

        if optimizer_idx == 0 or optimizer_idx is None:
            # Train generator
            self.opt_g.zero_grad()
            import torchvision.utils as vutils
            # Generate a batch of images
            z = torch.randn(batch_size, self.hparams.latent_dim)
            gen_imgs = self.generator(z, labels)
            vutils.save_image(gen_imgs, "generated_images1.png", normalize=True)

            # Calculate generator loss
            g_loss = self.adversarial_loss(self.discriminator(gen_imgs,labels), valid)

            # Backward pass and optimize
            g_loss.requires_grad_(True)
            g_loss.backward()
            self.opt_g.step()

            log_dict["g_loss"] = g_loss.item()

        if optimizer_idx == 1 or optimizer_idx is None:
            # Train discriminator
            self.opt_d.zero_grad()

            # Real images
            real_loss = self.adversarial_loss(self.discriminator(imgs,labels), valid)

            # Fake images
            z = torch.randn(batch_size, self.hparams.latent_dim)
            gen_imgs = self.generator(z, labels)
            fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach(),labels), fake)

            # Total discriminator loss
            d_loss = (real_loss + fake_loss) / 2
            d_loss.requires_grad_(True)
            # Backward pass and optimize
            d_loss.backward()
            self.opt_d.step()

            log_dict["d_loss"] = d_loss.item()

        return log_dict, loss

    def on_epoch_end(self):
    # Generate fake images
        z = torch.randn(10, self.hparams.latent_dim)
        labels = torch.randint(0, 10, (10,))
        gen_imgs = self.generator(z, labels)



    # Log fake images to Wandb
    #    for logger in self.trainer.logger:
    #        if type(logger).__name__ == "WandbLogger":
    #            logger.experiment.log({"gen_imgs": [wandb.Image(img) for img in gen_imgs]})

    def test_dataloader(self):
        # 返回测试数据集的数据加载器
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)