import logging
from dataclasses import dataclass
from pathlib import Path
from typing import *

import torch
import torch.nn as nn
from torch.optim import Optimizer

from ste_gan.models.generator import EMGGenerator


def fix_state_dict(state_dict):
    new_state_dict = OrderedDict()
    prefix = '_orig_mod.' 
    for sd_key in state_dict.keys():
        if prefix in sd_key:
            new_state_dict[sd_key.replace(prefix, "")] = state_dict[sd_key]
        else:
            new_state_dict[sd_key] = state_dict[sd_key]
    return new_state_dict

def load_latest_checkpoint(
    checkpoint: Path,
    device: torch.device,
    generator: nn.Module,
    discriminator: nn.Module
) -> Tuple[nn.Module, nn.Module]:
    epochs = []
    for f in checkpoint.glob('checkpoint-*.pt'):
        try:
            epoch_int = int(f.stem.split('-')[1])
            epochs.append(epoch_int)
        except ValueError:
            pass
    epochs.sort()
    latest = f'{epochs[-1]:08d}'
    
    logging.info(f"LOADING GENERATOR CHECKPOINT: {checkpoint / f'netG-{latest}.pt'}")
    generator.load_state_dict(
        fix_state_dict(torch.load(checkpoint / f'netG-{latest}.pt', map_location=device)))
    logging.info(f"LOADING DISCRIMINATOR CHECKPOINT: {checkpoint / f'netD-{latest}.pt'}")
    discriminator.load_state_dict(
        fix_state_dict(torch.load(checkpoint / f'netD-{latest}.pt', map_location=device)))
    
    
    return (
        generator, discriminator
    )


def initialize_emg_generator(
    generator: nn.Module,
    gen_ckpt_file_path: Path,
    device: torch.device,
) -> EMGGenerator:
    generator.load_state_dict(
        torch.load(gen_ckpt_file_path, map_location=device)
    )
    generator.eval()
    return generator
