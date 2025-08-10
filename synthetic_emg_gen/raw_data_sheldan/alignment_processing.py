#!/usr/bin/env python3


import librosa as lib
import numpy as np
import os
import pickle
import random
import shutil
import subprocess
from tqdm import tqdm
from datetime import datetime
# Get the current date and time
print(datetime.now())
pwd = os.getcwd()
root_file_path=os.path.abspath(os.path.dirname(pwd)+os.path.sep)

def get_file_list(path, fileType):
    file_list = []
    for filename in os.listdir(path):
        # 如果是文件夹，则递归处理
        if os.path.isdir(os.path.join(path, filename)):
            file_list.extend(get_file_list(os.path.join(path, filename), fileType))
        elif os.path.isfile(os.path.join(path, filename)):
            if filename.endswith(fileType):
                file_list.append(os.path.join(path, filename))
    return file_list


def generate_separate_transcript_files(dataset_path):
    path = root_file_path +str(dataset_path)
    file_names = get_file_list(path, 'trans.txt')

    for file_name in file_names:
        with open(file_name, 'r') as file:
            # Read all lines from the file
            lines = file.readlines()

        parent_dirs = os.path.dirname(file_name)
        
        # Process each line
        for line in lines:
            # Split the line at the first space to separate the identifier from the sentence
            parts = line.split(' ', 1)
        
            assert len(parts) == 2

            identifier, sentence = parts[0], parts[1]
    
            # Create a new filename based on the identifier
            output_file_name = f"{parent_dirs}/{identifier}.txt"
    
            # Write the sentence to the new file
            with open(output_file_name, 'w') as output_file:
                output_file.write(sentence)
    
            print(f"Saved: {output_file_name}")



def run_mfa_align_one(sound_file_path, text_file_path, dictionary_path, acoustic_model_path, output_path, options=''):

    # Construct the command
    command = f"mfa align_one {options} {sound_file_path} {text_file_path} {dictionary_path} {acoustic_model_path} {output_path}"

    try:
        # Run the command using subprocess
        subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        
    
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        print("Return code:", e.returncode)
        print("Output:", e.stdout)
        print("Errors:", e.stderr)

if __name__ == '__main__':
    
    generate_tscp_file = False
    if generate_tscp_file:
        dataset_path = './LibriSpeech'
        print('generate_tscp_file')
        generate_separate_transcript_files(dataset_path)

    run_alignment = True
    if run_alignment:
        print('run_alignment')
        '''load transcript'''
        AudioFileType = '.flac'
        TranscriptFileType = '.txt'


        dictionary_path = 'path/to/project/forced-aligner/MFA/pretrained_models/dictionary/english_us_arpa.dict'
        acoustic_model_path = 'path/to/project/forced-aligner/MFA/pretrained_models/acoustic/english_us_arpa.zip'

        output_path = './MFA_output'   
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)

        sound_dataset_path = './LibriSpeech/dev-clean'
        num_cores = os.cpu_count()
        print(f'{num_cores=}')
        command = f"mfa align  --output_format short_textgrid  --final_clean --clean --use_mp --num_jobs {num_cores} -v --no_quiet\
            {sound_dataset_path} {dictionary_path} {acoustic_model_path} {output_path}"
        
        try:
            # Run the command using subprocess
            subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
            
        
        except subprocess.CalledProcessError as e:
            print(f"An error occurred: {e}")
            print("Return code:", e.returncode)
            print("Output:", e.stdout)
            print("Errors:", e.stderr)

print('Done.')