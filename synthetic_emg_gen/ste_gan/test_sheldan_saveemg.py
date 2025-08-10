"""Main Test script for the STE-GAN from Sheldan.

Adapted from: https://github.com/descriptinc/cargan/blob/master/cargan/train.py
"""
import argparse
from datetime import datetime
import functools
import itertools
import logging
from pathlib import Path
import os
import random
import shutil
import sys
import time
from tqdm import tqdm

import numpy as np
import torch
import torch._dynamo
# torch._dynamo.config.suppress_errors = True

import torch.multiprocessing as mp
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

import ste_gan
from ste_gan.constants import DataType
from ste_gan.data_sheldan.emg_dataset import EMGDataset
from ste_gan.data_sheldan.loader import loaders_via_config
# from ste_gan.data.emg_dataset import EMGDataset
# from ste_gan.data.loader import loaders_via_config
from ste_gan.utils.plot_utils import plot_real_vs_fake_emg_signal_with_envelope
from ste_gan.losses.emg_encoder_loss import (EMGEncoderLoss,
                                             EMGEncoderLossOutput)
from ste_gan.losses.time_domain_loss import MultiTimeDomainFeatureLoss
from ste_gan.models.discriminator import init_emg_discriminators
from ste_gan.models.emg_encoder import load_emg_encoder
from ste_gan.models.generator import init_emg_generator
from ste_gan.train_utils import (add_eval_hyperparams_to_parser,
                                 create_ste_gan_model_name, load_config,
                                 mean_error, phoneme_accuracy,
                                 phoneme_accuracy_no_silence)
from ste_gan.utils.common import load_latest_checkpoint


def test(
    cfg: DictConfig,
    train_output_directory: Path,
    model_directory: Path, 
    torch_device: str,
    debug: bool,
    emg_enc_ckpt: Path = None,
    filter_threshold=0.5,
):
    print(f"Using filter threshold: {filter_threshold}")
    ###############
    # Load models #
    ###############
    device = torch.device(torch_device)

    netG = init_emg_generator(cfg)
    netD = init_emg_discriminators(cfg)

    checkpoint = train_output_directory
    logging.info(f"Loading checkpoint: {checkpoint}")
    netG, netD = load_latest_checkpoint(
        checkpoint, device, netG, netD
    )

    netG.to(device)
    netD.to(device)

    netG.eval()
    netD.eval()

    logging.info(f"Initializing EMG Encoder Model with EMG encoder checkpoint: {emg_enc_ckpt}")
    emg_encoder = load_emg_encoder(cfg, device, emg_enc_ckpt)
    emg_encoder.eval()
    emg_encoder.to(device)


    #######################
    # Create data loaders #
    #######################
    logging.info("Loading Data -- this can take a while")
    data_root = Path(cfg.data.dataset_root)
    logging.info(f"Data Set root: {data_root}")

    test_loader = loaders_via_config(cfg)
    # train_loader, valid_loader, test_loader = loaders_via_config(cfg)
    logging.info(f'{len(test_loader)=}')

    # train_data_set: EMGDataset = train_loader.dataset
    # val_data_set: EMGDataset = valid_loader.dataset
    test_data_set: EMGDataset = test_loader.dataset

    session_idx_to_id = test_data_set.save_session_and_speaking_mode_mapping_json(model_directory)

    #########
    # Test #
    #########
    logging.info(f"Starting Testing")
    log_start = time.time()

    '''initializing losses'''
    logging.info(f"Initializing Losses")
    multi_td_loss = MultiTimeDomainFeatureLoss(cfg.data.num_emg_channels).to(device)
    emg_encoder_loss = EMGEncoderLoss(emg_encoder).to(device)

    if int(torch.__version__[0]) >= 2:
        # logging.info(f"Compiling models...PyTorch version: {torch.__version__}")
        multi_td_loss = torch.compile(multi_td_loss)
        emg_encoder_loss = torch.compile(emg_encoder_loss)
    else:
        logging.warning(f"Will NOT compile models. Torch version: {torch.__version__}")
    

 
    speech_feature_type = cfg.model.speech_feature_type
    print(f'{speech_feature_type=}')

    ######################
    # Starting Testing #
    ######################
    with torch.no_grad():  # No gradient calculation during evaluation
        logging.info(f"Starting Testing")
        test_start = time.time()

        su_errors = []
        phoneme_errors = []

        # Phoneme accuracy including silences
        test_num_phones = 0
        test_num_phones_correct = 0
        # Phoneme Accuracy computation with ignoring silences
        test_num_silence = 0
        test_num_phones_correct_no_silence = 0

        '''create save emg data path'''
        result_path = './raw_data_sheldan/emg_generated'
        if os.path.exists(result_path):
            shutil.rmtree(result_path)
        os.makedirs(result_path, exist_ok=True)

        for i, feat_batch in enumerate(test_loader):
            # x_t = feat_batch[DataType.REAL_EMG].to(device)
            s_t = feat_batch[speech_feature_type].to(device)
            assert speech_feature_type == DataType.SPEECH_UNITS
            speech_units_t = s_t

            spk_mode_idx = feat_batch[DataType.SPEAKING_MODE_INDEX].to(device)
            sess_idx = feat_batch[DataType.SESSION_INDEX].to(device)
            phoneme_targets = feat_batch[DataType.PHONEMES].to(device)

            '''save generated EMG'''
            file_id = feat_batch[DataType.FILE_IDS].to(device).cpu().int().item()
            tmp = (session_idx_to_id[sess_idx.cpu().int().item()].split("_"))[-1]
            emg_save_name = f'{tmp}-{file_id:04}'
            '''sheldan: RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED'''
            # sess_idx = sess_idx.clamp(0, cfg.data.num_emg_sessions-1)
            sess_idx = sess_idx % cfg.data.num_emg_sessions
            file_path = os.path.join(result_path, f'{emg_save_name}.npy')
 
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    pred_emg = netG.generate(s_t, sess_idx, spk_mode_idx).squeeze(0).detach().cpu().numpy()
                    x_pred_t = netG(s_t, sess_idx, spk_mode_idx)

                    # wave_errors.append(torch.nn.functional.mse_loss(x_pred_t, x_t).item())
                    # td_errors.append(multi_td_loss(x_t, x_pred_t).item())
                    # print(f'{x_pred_t.size()=}, {speech_units_t.size()=}, {phoneme_targets.size()=}')
                    emg_enc_loss_output = emg_encoder_loss(x_pred_t, speech_units_t, phoneme_targets)
                
                    su_errors.append(emg_enc_loss_output.speech_unit_loss.item())
                    phoneme_errors.append(emg_enc_loss_output.phoneme_loss.item())
                    # print(f'{emg_enc_loss_output.speech_unit_loss.item()=}')
                    # print(f'{emg_enc_loss_output.phoneme_loss.item()=}')

                    if not emg_enc_loss_output.phoneme_loss.item() > filter_threshold:
                        np.save(file_path, pred_emg)

                    logging.info(f'Iteration: {i}, su_errors: {emg_enc_loss_output.speech_unit_loss.item()}')
                    logging.info(f'Iteration: {i}, phoneme_errors: {emg_enc_loss_output.phoneme_loss.item()}')
            
            test_num_phones += emg_enc_loss_output.num_phones
            test_num_phones_correct += emg_enc_loss_output.num_correct_phones
            
            test_num_silence += emg_enc_loss_output.num_silence_phones
            test_num_phones_correct_no_silence += emg_enc_loss_output.num_correct_phones_no_silence


            
        # Calculate Mean Test errors
        # avg_test_td_error = mean_error(td_errors)
        # avg_test_wave_error = mean_error(wave_errors)
        avg_test_phoneme_error = mean_error(phoneme_errors)
        avg_su_error = mean_error(su_errors)
        avg_phoneme_accuracy = phoneme_accuracy(test_num_phones, test_num_phones_correct)
        avg_phoneme_accuracy_no_sil = phoneme_accuracy_no_silence(test_num_phones, 
                                                                    test_num_phones_correct_no_silence,
                                                                    test_num_silence)

        logging.info("-" * 100)
        logging.info("Took %5.4fs to run test" % (time.time() - test_start))
        logging.info(f"\t - Avg. Test Speech Unit Error : {avg_su_error}")
        logging.info(f"\t - Avg. Test Phoneme Error: {avg_test_phoneme_error}")
        logging.info(f"\t - Avg. Test Phoneme Accuracy: {avg_phoneme_accuracy}")
        logging.info(f"\t - Avg. Test Phoneme Accuracy (No Sil.): {avg_phoneme_accuracy_no_sil}")
        # logging.info(f"\t - Avg. Test Waveform Error: {avg_test_wave_error}")
        # logging.info(f"\t - Avg. Test Multi-TD Test Error: {avg_test_td_error}")
        logging.info("-" * 100)

    logging.info("Took %5.4fs to run program" % (time.time() - log_start))

###############################################################################
# Entry point
###############################################################################
def main(cfg: DictConfig, debug: bool, emg_enc_ckpt: Path, filter_threshold: float = 0.5, **kwargs):
    dataset_root = cfg.data.dataset_root
    print(f"Data root: {dataset_root}")
    print(f"Debug (argparse): {debug}")
    
    if not debug and cfg.train.debug:
        print(f"WARNING: SETTING GLOBAL DEBUG FLAG")
        debug = True
    
    # reDefine train result dir
    model_base_dir = Path(cfg.model_base_dir)
    train_output_directory  = model_base_dir / create_ste_gan_model_name(
        cfg, add_timestamp=False, debug=debug,
    )

    # Create output dir
    output_directory = model_base_dir / 'synthetic_emg_gen'
    if output_directory.exists():
        shutil.rmtree(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_directory}")

    done_file = output_directory / ".done"
    if output_directory and done_file.exists():
        logging.warning(f"Exiting training script as '.done' file exists: {done_file.absolute()}")
        sys.exit()

    # Save configuration
    config_file = output_directory / "config.yaml"
    if not config_file.exists():
        with open(config_file, '+w') as fp:
            OmegaConf.save(config=cfg, f=fp.name)

    logging.info(OmegaConf.to_yaml(cfg))
    logging.getLogger().setLevel(logging.INFO)
    log_file = output_directory / "log.txt"
    fh = logging.FileHandler(str(log_file.absolute()))
    logging.getLogger().addHandler(fh) 

    current_time = datetime.now()
    logging.info(f'current time is :{current_time}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test(cfg, train_output_directory, output_directory, device, debug, emg_enc_ckpt, filter_threshold=filter_threshold)
    

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ste_gan_base_gantts.yaml",
                        help="The main training configuration for this run.")
    parser.add_argument("--data", type=str, default="configs/data/sheldan_corpus.yaml",
                        help="A path to a data configuration file.")
    parser.add_argument("--emg_enc_cfg", type=str, default="configs/emg_encoder/conv_transformer.yaml",
                        help="A path to an EMG encoder configuration file.")
    parser.add_argument("--emg_enc_ckpt", type=str, default="exp/emg_encoder/EMGEncoderTransformer_voiced_only__seq_len__200__data_gaddy_complete/best_val_loss_model.pt",
                        help="A path to a checkpooint of a pre-trained EMG encoder. Must correspond to the EMG encoder configuration in 'emg_enc_cfg'.")
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Whether to run the training script in debug mode.')
    parser.add_argument(
    '--filter_threshold',
    type=float,
    default=0.5,
    help='Threshold value to filter data in the test function.'
    )
    
    parser = add_eval_hyperparams_to_parser(parser)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg = load_config(args)
    args.cfg = cfg
    main(**vars(args))
