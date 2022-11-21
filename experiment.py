
import os
import torch
from torch import Tensor
from torch import optim
import pytorch_lightning as pl
import torchvision.utils as vutils

from models.vanila_vae import VanillaVAE


class VAEXperiment(pl.LightningModule):

  def __init__(self, vae_model: VanillaVAE, params: dict) -> None:
    super(VAEXperiment, self).__init__()

    self.model = vae_model
    self.params = params
    self.curr_device = None
    self._is_first_val = True
    self._test_input = None
    self._test_label = None
    self.hold_graph = False
    try:
        self.hold_graph = self.params['retain_first_backpass']
    except:
        pass

  def forward(self, input: Tensor, **kwargs) -> Tensor:
    return self.model(input, **kwargs)


  def training_step(self, batch, batch_idx, optimizer_idx = 0):
    real_img, labels = batch
    self.curr_device = real_img.device

    results = self.forward(real_img, labels = labels)
    train_loss = self.model.loss_function(*results,
                                          M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,
                                          optimizer_idx=optimizer_idx,
                                          batch_idx = batch_idx)

    self.log_dict({key: val.item() for key, val in train_loss.items()})

    return train_loss['loss']


  def validation_step(self, batch, batch_idx, optimizer_idx = 0):
    real_img, labels = batch
    self.curr_device = real_img.device

    results = self.forward(real_img, labels = labels)
    val_loss = self.model.loss_function(*results,
                                        M_N = 1.0, #real_img.shape[0]/ self.num_val_imgs,
                                        optimizer_idx = optimizer_idx,
                                        batch_idx = batch_idx)

    self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)


  def _get_validation_batch(self):
    if self.params['random_val_batch'] == True:            
      return next(iter(self.trainer.datamodule.test_dataloader()))
    
    if self._is_first_val == True:
      self._is_first_val = False
      self._test_input, self._test_label = next(iter(self.trainer.datamodule.test_dataloader()))
      vutils.save_image(self._test_input,
                      os.path.join(self.logger.log_dir , 
                                    "Reconstructions", 
                                    f"recons_{self.logger.name}_original.png"),
                      normalize=True,
                      nrow=8) 
    
    return self._test_input, self._test_label 


  def on_validation_end(self) -> None:
    test_input, test_label = self._get_validation_batch()
    test_input = test_input.to(self.curr_device)
    test_label = test_label.to(self.curr_device)

    self.test_reconstruction(test_input, test_label)
    if (self.params['sample_size'] > 0):
      self.sample_images(test_input, test_label)


  def test_reconstruction(self, test_input, test_label) -> None:
    # Get sample reconstruction image

    recons = self.model.generate(test_input, labels = test_label)
    vutils.save_image(recons.data,
                      os.path.join(self.logger.log_dir , 
                                    "Reconstructions", 
                                    f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                      normalize=True,
                      nrow=12) 


  def sample_images(self, test_input, test_label):
    try:
      samples = self.model.sample(144,
                                  self.curr_device,
                                  labels = test_label)
      vutils.save_image(samples.cpu().data,
                        os.path.join(self.logger.log_dir , 
                                      "Samples",      
                                      f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                        normalize=True,
                        nrow=12)
    except Warning:
        pass

  def configure_optimizers(self):
    optims = []
    scheds = []

    optimizer = optim.Adam(self.model.parameters(),
                            lr=self.params['LR'],
                            weight_decay=self.params['weight_decay'])
    optims.append(optimizer)
    # Check if more than 1 optimizer is required (Used for adversarial training)
    try:
      if self.params['LR_2'] is not None:
          optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                  lr=self.params['LR_2'])
          optims.append(optimizer2)
    except:
        pass

    try:
      if self.params['scheduler_gamma'] is not None:
        scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                      gamma = self.params['scheduler_gamma'])
        scheds.append(scheduler)

      # Check if another scheduler is required for the second optimizer
      try:
          if self.params['scheduler_gamma_2'] is not None:
              scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                            gamma = self.params['scheduler_gamma_2'])
              scheds.append(scheduler2)
      except:
          pass
      return optims, scheds
    except:
        return optims