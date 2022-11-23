
import os
import torch
from torch import Tensor
from torch import optim
import pytorch_lightning as pl
import torchvision.utils as vutils

from models.resnet_ae import ResnetAutoencoder


class AEExperiment(pl.LightningModule):

  def __init__(self, vae_model: ResnetAutoencoder, params: dict) -> None:
    super(AEExperiment, self).__init__()

    self.model = vae_model
    self.params = params
    self.curr_device = None
    self._is_first_val = True
    self._recon_batch = None
    self.hold_graph = False
    try:
        self.hold_graph = self.params['retain_first_backpass']
    except:
        pass

  def forward(self, input: Tensor, **kwargs) -> Tensor:
    return self.model(input, **kwargs)


  def training_step(self, batch, batch_idx, optimizer_idx = 0):
    original, labels = batch
    self.curr_device = original.device

    recon = self.forward(original)
    train_loss = self.model.loss_function(original, recon,
                                          optimizer_idx=optimizer_idx,
                                          batch_idx = batch_idx)

    self.log('loss', train_loss, on_epoch=True, on_step=False, sync_dist=True)

    return train_loss


  def validation_step(self, batch, batch_idx, optimizer_idx = 0):
    original, labels = batch
    self.curr_device = original.device

    recon = self.forward(original)
    val_loss = self.model.loss_function(original, recon,
                                        optimizer_idx = optimizer_idx,
                                        batch_idx = batch_idx)

    self.log('val_loss', val_loss, on_epoch=True, on_step=False, sync_dist=True)


  def _get_recon_batch(self):
    # if self.params['random_val_batch'] == True:            
    #   return next(iter(self.trainer.datamodule.test_dataloader()))
    
    if self._is_first_val == True:
      self._is_first_val = False
      full_recon_batch, _ = next(iter(self.trainer.datamodule.test_dataloader()))
      self._recon_batch = full_recon_batch[:16]
      vutils.save_image(self._recon_batch,
                      os.path.join(self.logger.log_dir , 
                                    "Reconstructions", 
                                    f"original_{self.logger.name}.png"),
                      normalize=True,
                      nrow=4) 
    
    return self._recon_batch 


  def on_validation_end(self) -> None:
    recon_batch = self._get_recon_batch()
    recon_batch = recon_batch.to(self.curr_device)

    self.test_reconstruction(recon_batch)


  def test_reconstruction(self, originals) -> None:
    recons = self.model.forward(originals)
    vutils.save_image(recons.data,
                      os.path.join(self.logger.log_dir , 
                                    "Reconstructions", 
                                    f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                      normalize=True,
                      nrow=4)


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