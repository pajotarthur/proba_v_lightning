"""
Runs a model on a single node across N-gpus.
"""
import os
from argparse import ArgumentParser

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger

from lg_model import ProbaModel

SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)


def main(hparams):
    """
    Main training routine specific for this project
    :param hparams:
    """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = ProbaModel(hparams)

    logger = TestTubeLogger(
            save_dir='logs/logger',
            name="grid_search_2",
            debug=False,
            create_git_tag=True,
            )

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = Trainer(
            gpus=hparams.gpus,
            logger=logger,
            # max_nb_epochs=200,
            min_nb_epochs=200,
            )

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)

    # gpu args
    parent_parser.add_argument(
            '--gpus',
            type=int,
            default=2,
            help='how many gpus'
            )

    parent_parser.add_argument(
            '--use_16bit',
            dest='use_16bit',
            action='store_true',
            help='if true uses 16 bit precision'
            )

    # each LightningModule defines arguments relevant to it
    parser = ProbaModel.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hyperparams)
