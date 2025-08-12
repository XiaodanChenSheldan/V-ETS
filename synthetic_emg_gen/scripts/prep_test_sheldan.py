import argparse
import json
import os
import pickle
import random
import re
from functools import lru_cache, partial
from pathlib import Path
from typing import *


from concurrent.futures import ThreadPoolExecutor
from typing import Any
import torch
import multiprocessing

import numpy as np
import soundfile as sf
import shutil
import torch.cuda
import torch.nn.functional as F
import tqdm

from ste_gan.constants import PHONEME_INVENTORY
from ste_gan.utils.audio_utils import (
    MFCCsCalculator, cut_audio_to_soft_speech_match_unit_frame_rate,
    read_phonemes)


def get_utterance_file_id_from_sample_dict(sample_dict: Dict) -> str:
    utt_idx = sample_dict['index']
    session_id = sample_dict['session_ids']
    silent_str = "normal"

    return f"voiced_parallel_data_{session_id}__{utt_idx}__{silent_str}"



def load_utterance(directory_info, voiced_data_locations, text_align_directory=None, hubert=None, device=None):
    assert hubert is not None
    assert device is not None
    
    directory_info_, index = directory_info
    directory_info1, directory_info2 = directory_info_.split('-')
    text_file_name = f"{directory_info1}/{directory_info2}/{directory_info_}-{index:04}.txt" 
    text_path = Path(os.path.join(voiced_data_locations, text_file_name))
    if not text_path.exists():
        raise ValueError(f"Text path does not exist: {text_path}")
    with open(text_path, 'r') as file:
        # Read all lines from the file
        lines = file.readlines()
    assert len(lines) == 1
    text = lines[0].strip()


    # audio_file_name = f"{directory_info1}/{directory_info2}/{directory_info_}-{index:04}.flac" 
    audio_file_name = f"{directory_info1}/{directory_info2}/{directory_info_}-{index:04}_clean.flac" 
    audio_path = Path(os.path.join(voiced_data_locations, audio_file_name))
    if not audio_path.exists():
        raise ValueError(f"Audio path does not exist: {audio_path}")
    
    audio, sr = sf.read(audio_path)
    assert sr == 16_000, "Audio must be sampled to 16kHz"
    
    audio = cut_audio_to_soft_speech_match_unit_frame_rate(audio)
    mfccs_calc = MFCCsCalculator()
    audio_for_mfccs = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).to(torch.float32)
    mfccs = mfccs_calc(audio_for_mfccs)
    mfccs = mfccs.squeeze().T.numpy()
    
    # Calculate soft speech units  speech units from audio
    audio_t = np.expand_dims(np.expand_dims(audio, 0), 0)
    audio_t = torch.from_numpy(audio_t).float().to(device)
    with torch.no_grad():
        speech_units = hubert.units(audio_t).squeeze().detach().cpu().numpy()
    del audio_t
    torch.cuda.empty_cache()

    if len(mfccs) % 2 == 1:
        mfccs = mfccs[:-1]  # Delete last element so that we have acoustic feats divisable by 2
    speech_units = speech_units[:(len(mfccs) // 2)]
    mfccs = mfccs[:2 * len(speech_units)]  

    tg_file_name = f"{directory_info1}/{directory_info2}/{directory_info_}-{index:04}.TextGrid" 
    tg_fname = Path(os.path.join(text_align_directory, tg_file_name))

    assert os.path.exists(tg_fname)
    phonemes = read_phonemes(tg_fname, speech_units.shape[0])

    # if os.path.exists(tg_fname):
    #     phonemes = read_phonemes(tg_fname, speech_units.shape[0])
    # else:
    #     if speech_units is not None:
    #         phonemes = np.zeros(speech_units.shape[0], dtype=np.int64) + PHONEME_INVENTORY.index('sil')
    #     else:
    #         phonemes = np.zeros(mfccs.shape[0] // 2, dtype=np.int64) + PHONEME_INVENTORY.index('sil')


    return mfccs, phonemes, speech_units, text, audio, audio_path


def only_include_alphanumeric_chars(text: str):
    return re.sub(r'\W+', '', text.strip())


class GenerateEMGDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        text_align_directory=None,
        voiced_data_locations=None,
        hubert=None,
        device=None
    ):
        for var in text_align_directory, hubert, device:
            assert var is not None

        assert text_align_directory is not None
        self.mfcc_calc = MFCCsCalculator()
        self.hubert = hubert
        self.device = device
        
        self.text_align_directory = text_align_directory
        self.voiced_data_locations = voiced_data_locations


        '''仅仅获取文件名'''
        fileType = 'TextGrid'
        self.example_indices = []
        for dirpath, dirnames, filenames in os.walk(self.text_align_directory):
            if not dirnames and filenames[0].endswith(fileType):
                for filename in filenames:
                    filename_split = filename.split('.')[0]
                    # Extract directory and session info
                    directory_info_, idx_str_ = filename_split.rsplit('-', 1)
                    # Add to session groups
                    self.example_indices.append((directory_info_, int(idx_str_)))


        self.num_sessions = len(self.example_indices)

    def __len__(self):
        return len(self.example_indices)


    @lru_cache(maxsize=None)
    def __getitem__(self, i):

        directory_info, idx = self.example_indices[i]

        mfccs, phonemes, speech_units, text, audio, audio_path \
        = load_utterance(self.example_indices[i], self.voiced_data_locations, self.text_align_directory,
                         hubert=self.hubert, device=self.device)
        

        session_ids = directory_info

        audio_file = str(audio_path)

        speech_units = torch.from_numpy(speech_units) if speech_units is not None else None
        result = {
            'audio': torch.from_numpy(audio),
            'mfccs': torch.from_numpy(mfccs),
            'speech_units': speech_units,
        }
        

        result['phonemes'] = torch.from_numpy(
            phonemes).pin_memory()  # either from this example if vocalized or aligned example if silent
        result['audio_file'] = audio_file
        result['index'] = idx
        result['text'] = text
        result['session_ids'] = session_ids

        
        return result

def save_samples_of_data_set(
    generate_emg_dataset: GenerateEMGDataset,
    root_path: Path,
    dry_run: bool = False,
):
    
    num_samples = len(generate_emg_dataset)

    used_split_name = "test" 
    save_sheldan_subset_dir  = root_path / used_split_name
    if os.path.exists(save_sheldan_subset_dir):
        shutil.rmtree(save_sheldan_subset_dir)
    os.makedirs(save_sheldan_subset_dir, exist_ok=True)
    for sample in tqdm.tqdm(generate_emg_dataset, total=num_samples, desc='In save_samples_of_data_set'):
        utt_file_id = get_utterance_file_id_from_sample_dict(sample)

        phonemes = sample["phonemes"]
        
        units = sample["speech_units"]
        mfccs = sample["mfccs"]
        audio = sample["audio"]


        if len(mfccs) % 2 == 1:
            mfccs = mfccs[:-1]  # Delete last element so that we have acoustic feats divisable by 2
        units = units [:(len(mfccs) // 2)]
        mfccs = mfccs[:2 * len(units)]  # Speech units are half the number of samples as acoustic feats -->
        
        assert len(units) == len(phonemes)


        save_sheldan_subset_dir.mkdir(exist_ok=True, parents=True)
        
        for sub_dir_name, data in zip(
            ["phonemes", "units", "mfccs"],
            [phonemes, units, mfccs]
        ):
            sub_dir = save_sheldan_subset_dir / sub_dir_name
            file_path = sub_dir / f"{utt_file_id}.pt"
            # print(f"Saving data of shape {data.shape} to: {file_path}")
            # print(f"{sub_dir_name} -- {data.shape} -> {file_path.absolute()}")
            if not dry_run:
                sub_dir.mkdir(exist_ok=True, parents=True)
                torch.save(data, file_path)

        # Save transcriptions
        transcriptions = sample["text"]
        sub_dir = save_sheldan_subset_dir / "transcriptions"
        file_path = sub_dir / f"{utt_file_id}.txt"
        # print(f"{transcriptions} -> {file_path.absolute()}")
        if not dry_run:
            sub_dir.mkdir(exist_ok=True, parents=True)
            with open(file_path, '+w') as fp:
                fp.write(transcriptions)
                
        # Save audio
        audio_save_path = save_sheldan_subset_dir / "audio" / f"{utt_file_id}.wav"
        # print(f"audio -- {audio.shape} -> {audio_save_path}")
        if not dry_run:
            audio_save_path.parent.mkdir(exist_ok=True, parents=True)
            sf.write(audio_save_path, audio.numpy(), samplerate=16_000)
      


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_alignment_dir", type=Path,
                        default=Path("raw_data_sheldan/MFA_output/"))
    parser.add_argument("--audio_root_dir", type=Path,
                        default=Path("raw_data_sheldan/LibriSpeech/dev-clean"))
    parser.add_argument("--target_dir", type=Path,
                        default=Path("data/sheldan_complete"))
    parser.add_argument("--unit_sr", type=int, default=50)
    parser.add_argument("--dry_run", type=bool, default=False)
    
    args = parser.parse_args()
    target_dir = Path(args.target_dir)
    target_dir.mkdir(exist_ok=True)


    # Load data sets
    # Setup train / dev / test sets
    # Only test here

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ## Init Soft HuBERT
    hubert = torch.hub.load("bshall/hubert:main", "hubert_soft", ).to(device)

    data_set_all = GenerateEMGDataset(
        text_align_directory=args.text_alignment_dir,
        voiced_data_locations=args.audio_root_dir,
        hubert=hubert,
        device=device,
    ) 

    # data_set_all = init_data
    print(f"Test Data set size: {len(data_set_all)}")

    save_samples_of_data_set(
        data_set_all, target_dir,
        # soft_speech_units_sample_rate=args.unit_sr,
        dry_run=args.dry_run
    )
    print('Done.')




if __name__ == "__main__":
    import sys
    main()
