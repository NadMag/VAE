import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def filter_null_by_cols(df, col_names):
  indcies = df.loc[:,col_names].notnull().all(1)
  return df.loc[indcies, col_names]


def plot_vae_loss(epochs, train_loss_by_e, val_loss_by_e):
  fig, (ax1, ax2) = plt.subplots(1,2)
  ax1.plot(epochs, train_loss_by_e.loc[:,'loss'], linestyle='dashed', color='r')
  ax1.plot(epochs, train_loss_by_e.loc[:,'recon_loss'])
  ax1.plot(epochs, train_loss_by_e.loc[:,'kld'] * 0.00025)
  ax2.plot(epochs, val_loss_by_e.loc[:,'val_loss'], linestyle='dashed', color='r')
  ax2.plot(epochs, val_loss_by_e.loc[:,'val_recon_loss'])
  ax2.plot(epochs, val_loss_by_e.loc[:,'val_kld'] * 0.00025)
  for ax in [ax1, ax2]:
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(["Loss", "Reconstruction L2", "Weighted KLD"],loc='upper right')

  fig.suptitle('Vanila VAE Loss By Epoch', fontsize=14)
  ax1.set_title(f'Train Loss')
  ax2.set_title(f'Validation Loss')

  fig.set_size_inches(12, 5)
  fig.subplots_adjust(wspace=0.3)
  plt.savefig('./logs/VanillaVAE/version_0/acc_by_scaling_factor.png')
  plt.show()


def plot_ae_loss(num_epochs, train_loss_by_e, val_loss_by_e, metrics_file_dir):
  epochs = range(num_epochs)
  fig, ax = plt.subplots()
  ax.plot(epochs, train_loss_by_e.loc[:,'loss'])
  ax.plot(epochs, val_loss_by_e.loc[:,'val_loss'])
  ax.set_xlabel('Epoch')
  ax.set_ylabel('Reconstruction Loss')
  ax.legend(["Train Loss", "Validation Loss"],loc='upper right')

  fig.suptitle('Resnet Autoencoder Loss By Epoch', fontsize=14)

  fig.set_size_inches(8, 5)
  plt.savefig(f'{metrics_file_dir}/loss_by_epoch.png')
  plt.show()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Creates graphs')
  parser.add_argument('--metrics_file_dir',  '-p',
                      dest="metrics_file_dir",
                      metavar='FILE',
                      help =  'path to the metrics file directory')
  
  parser.add_argument('--vae', action='store_true')

  args = parser.parse_args()
  loss_by_e = pd.read_csv(f'{args.metrics_file_dir}/metrics.csv')
  train_col_names = ['loss', 'recon_loss', 'kld'] if args.vae else ['loss']
  val_col_names = [f"val_{name}" for name in train_col_names]
  train_loss_by_e = filter_null_by_cols(loss_by_e, train_col_names)
  val_loss_by_e = filter_null_by_cols(loss_by_e, val_col_names)
  if args.vae:
    plot_vae_loss(len(train_loss_by_e), train_loss_by_e, val_loss_by_e)
  else:
    plot_ae_loss(len(train_loss_by_e), train_loss_by_e, val_loss_by_e, args.metrics_file_dir)