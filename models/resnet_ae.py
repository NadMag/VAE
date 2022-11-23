#https://pytorch-lightning.readthedocs.io/en/stable/notebooks/course_UvA-DL/08-deep-autoencoders.html
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class ResidualLayer(nn.Module):

  def __init__(self, in_channels: int, use_bn: bool = True):
    super(ResidualLayer, self).__init__()
    layers = [      
      nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
      nn.LeakyReLU(True)
    ]
    if use_bn:
      layers.insert(1, nn.BatchNorm2d(in_channels))
    
    self.resblock = nn.Sequential(*layers)

  def forward(self, input: Tensor) -> Tensor:
    return input + self.resblock(input)


class ResnetEncoder(nn.Module):
  def __init__(self, num_input_channels: int, hidden_dims: List, 
  latent_dim: int, use_bn: bool = True):
      """
      Args:
          num_input_channels : Number of input channels of the image.
          hidden_dims :  
          latent_dim : Dimensionality of latent representation z
          act_fn : Activation function used throughout the encoder network
      """
      super().__init__()
      if hidden_dims is None:
        hidden_dims = [16, 32, 64, 128, 256, 512] #256 => 4
      self.hidden_dims = hidden_dims

      modules = []
      # Build Encoder
      in_channels = self.add_downsample_blocks(num_input_channels, 0, 3, use_bn, modules)
      modules.append(ResidualLayer(in_channels, use_bn))
      in_channels = self.add_downsample_blocks(in_channels, 3, 5, use_bn, modules)
      modules.append(ResidualLayer(in_channels))
      self.add_downsample_blocks(in_channels, 5, 6, use_bn, modules)
      modules.append(nn.Flatten())
      modules.append(nn.Linear(self.hidden_dims[-1]*16, latent_dim))
      self.net = nn.Sequential(*modules)

  def add_downsample_blocks(self, num_input_channels, start, end, use_bn, modules):
      in_channels = num_input_channels
      for i in range(start, end):
        layers = [
          nn.Conv2d(in_channels,
                        self.hidden_dims[i],
                        kernel_size=3,
                        stride = 2,
                        padding=1),
          nn.LeakyReLU()
        ]
        if use_bn:
          layers.insert(1, nn.BatchNorm2d(self.hidden_dims[i]))

        modules.append(nn.Sequential(*layers))
        in_channels = self.hidden_dims[i]

      return in_channels

  def forward(self, x):
      return self.net(x)


class ResnetDecoder(nn.Module):
    def __init__(self, num_output_channels: int, hidden_dims: List, 
      latent_dim: int, use_bn: bool = True):
        """
        Args:
           num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
           base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        if hidden_dims is None:
          hidden_dims = [16, 32, 64, 128, 256, 512].reverse() #256 => 4
        self.hidden_dims = hidden_dims
        
        self.linear = nn.Linear(latent_dim, hidden_dims[0]*16)
        modules = []
        self.add_upsample_blocks(0, 3, use_bn, modules)
        modules.append(ResidualLayer(hidden_dims[3], use_bn))
        self.add_upsample_blocks(3,5, use_bn, modules)
        modules.append(ResidualLayer(hidden_dims[5], use_bn))
        modules.append(
            nn.Sequential(
              nn.ConvTranspose2d(hidden_dims[-1],
                          num_output_channels,
                          kernel_size=3,
                          stride = 2,
                          padding=1,
                          output_padding=1),
              nn.Tanh())
        )
        self.net = nn.Sequential(*modules)

    def add_upsample_blocks(self, start, end, use_bn, modules):
      for i in range(start, end):
        layers = [
            nn.ConvTranspose2d(self.hidden_dims[i],
              self.hidden_dims[i + 1],
              kernel_size=3,
              stride = 2,
              padding=1,
              output_padding=1),
            nn.LeakyReLU()
        ]
        if use_bn:
          layers.insert(1, nn.BatchNorm2d(self.hidden_dims[i+1]))

        modules.append(nn.Sequential(*layers))

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x


class ResnetAutoencoder(nn.Module):
    def __init__(self,
                in_channels: int,
                latent_dim: int,
                hidden_dims: List = None,
                use_bn: bool = True,
                **kwargs) -> None:

      super(ResnetAutoencoder, self).__init__()
      self.latent_dim = latent_dim
      if hidden_dims is None:
        hidden_dims = [16, 32, 64, 128, 256, 512] #256 => 4
      self.hidden_dims = hidden_dims
      
      self.encoder = ResnetEncoder(in_channels, self.hidden_dims, self.latent_dim, use_bn)
      self.decoder = ResnetDecoder(in_channels, list(reversed(self.hidden_dims)), self.latent_dim, use_bn)

    def encode(self, input: Tensor) -> List[Tensor]:
      """
      Encodes the input by passing through the encoder network
      and returns the latent codes.
      :param input: (Tensor) Input tensor to encoder [N x C x H x W]
      :return: (Tensor) List of latent codes
      """
      return self.encoder(input)

    def decode(self, z: Tensor) -> Tensor:
      return self.decoder(z)

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
      latent = self.encode(input)
      recon = self.decode(latent)
      return recon

    def loss_function(self,
                    original,
                    recon,
                    *args,
                    **kwargs) -> dict:
      """
      Computes the VAE loss function.
      KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
      :param args:
      :param kwargs:
      :return:
      """

      recons_loss =F.mse_loss(recon, original)
      return recons_loss