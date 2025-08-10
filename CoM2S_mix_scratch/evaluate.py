import sys
import os
import logging

import tqdm
import shutil

import torch
from torch import nn

from architecture import Model
from transduction_model_sheldan_cont import test, save_output
from read_emg_sheldan import EMGDataset
from asr_evaluation import evaluate
from data_utils import phoneme_inventory, print_confusion
from vocoder import Vocoder

from time import process_time
from datetime import datetime

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_list('models', [], 'identifiers of models to evaluate')
flags.DEFINE_boolean('dev', False, 'evaluate dev insead of test')

class EnsembleModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x, x_raw, sess):
        ys = []
        ps = []
        for model in self.models:
            y, p = model(x, x_raw, sess)
            ys.append(y)
            ps.append(p)
        return torch.stack(ys,0).mean(0), torch.stack(ps,0).mean(0)

def main():

    dev = FLAGS.dev
    testset = EMGDataset(dev=dev, test=not dev)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_start = process_time()
    models = []
    for fname in FLAGS.models:
        state_dict = torch.load(fname, weights_only=True)
        logging.info(f'***\n{testset.num_features=}, {testset.num_speech_features=}, {testset.num_emg=}, {len(testset)=}\n***')
        model = Model(testset.num_features, testset.num_speech_features, len(phoneme_inventory)).to(device)
        model.load_state_dict(state_dict)
        models.append(model)
    ensemble = EnsembleModel(models)
    model_end = process_time()
    logging.info(f'Model start time: {model_start}')
    logging.info(f'Model end time: {model_end}')
    logging.info(f'Model elapsed time in seconds: {model_end-model_start}')

    _, _, confusion = test(ensemble, testset, device)
    print_confusion(confusion)

    vocoder = Vocoder()
    logging.info(f'SHELDAN: {len(testset)=}')
    logging.info(f'SHELDAN: {testset.exp_idx=}')

    for i, datapoint in enumerate(tqdm.tqdm(testset, 'Generate outputs', disable=None)):
        save_output(ensemble, datapoint, os.path.join(FLAGS.output_directory, f'example_output_{i}.wav'), device, testset.mfcc_norm, vocoder)

    evaluate(testset, FLAGS.output_directory)

if __name__ == "__main__":
    FLAGS(sys.argv)

    if os.path.exists(FLAGS.output_directory):
        shutil.rmtree(FLAGS.output_directory)
    os.makedirs(FLAGS.output_directory)
    logging.basicConfig(handlers=[
            logging.FileHandler(os.path.join(FLAGS.output_directory, 'eval_log.txt'), 'w'),
            logging.StreamHandler()
            ], level=logging.INFO, format="%(message)s")
    
    current_time = datetime.now()
    logging.info(f'Current time: {current_time}')
    time_start = process_time() 
    logging.info(f'Whole program start time: {time_start}')
    main()
    time_end = process_time() 
    logging.info(f'Whole program end time: {time_end}')
    logging.info(f'Whole program elapsed time in seconds: {time_end-time_start}')
