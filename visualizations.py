import os
import yaml
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
import argparse

def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha,wa = imga.shape[:2]
    hb,wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa+wb
    new_img = np.zeros(shape=(max_height, total_width, 3))
    new_img[:ha,:wa]=imga
    new_img[:hb,wa:wa+wb]=imgb
    return new_img

def create_reconstruction_ani(recon_dir, model_name, epochs, skip_epochs):
    fig = plt.figure(figsize=(10,5))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    plt.axis("off")
    frames = []

    org = mpimg.imread(f'{recon_dir}/original_{model_name}.png')
    for i in range(0, epochs, skip_epochs):
      recon = mpimg.imread(f'{recon_dir}/recons_{model_name}_Epoch_{i}.png')
      comp = concat_images(org, recon)
      frames.append([plt.imshow(comp, animated=True)])

    ani = animation.ArtistAnimation(fig, frames, interval=500, repeat_delay=1500, blit=True)
    writergif = animation.PillowWriter(fps=1) 
    ani.save(f'{recon_dir}/recon.gif', writer=writergif)


def main():
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config',  '-c',
                      dest="filename",
                      metavar='FILE',
                      help =  'path to the config file',
                      default='configs/vae.yaml')
    parser.add_argument('--skip',
                  dest="skip_epochs",
                  help =  'save every n epoch',
                  type=int,
                  default='2')
    parser.add_argument('--version',
                  dest="version",
                  help = 'model version',
                  default='0')

    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
  
    logging_params = config["logging_params"]
    recon_dir = f'./{logging_params["save_dir"]}/{logging_params["name"]}/version_{args.version}/Reconstructions'
    create_reconstruction_ani(
      recon_dir,
      logging_params["name"], 
      config['trainer_params']['max_epochs'], 
      args.skip_epochs
    )


if __name__ == "__main__":
  main()