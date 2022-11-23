import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models.resnet_ae import ResnetAutoencoder
from ae_expirement import AEExperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from dataset import VAEDataset

def main():
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the config file',
                        default='configs/vae.yaml')

    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)


    tb_logger =  CSVLogger(save_dir=config['logging_params']['save_dir'],
                                name=config['model_params']['name'])

    # For reproducibility
    seed_everything(config['exp_params']['manual_seed'], True)

    data = VAEDataset(**config["data_params"])

    data.setup()
    print(len(data.train_dataset))
    print(len(data.val_dataset))
    # model = vae_models[config['model_params']['name']](**config['model_params'])
    model = ResnetAutoencoder(**config['model_params'])
    experiment = AEExperiment(model,
                            config['exp_params'])

    runner = Trainer(logger=tb_logger,
                    strategy=DDPStrategy(find_unused_parameters=False),
                    **config['trainer_params'])


    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment, datamodule=data)

if __name__ == "__main__":
    main()