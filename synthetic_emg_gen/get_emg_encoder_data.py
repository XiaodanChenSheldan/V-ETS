#!/usr/bin/env python3


import librosa as lib
import numpy as np
import json
import os
import pickle
import random
import shutil
import subprocess
from tqdm import tqdm

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

def get_file_name_list(path, fileType):
    file_list = []
    for filename in os.listdir(path):
        # 如果是文件夹，则递归处理
        if os.path.isdir(os.path.join(path, filename)):
            file_list.extend(get_file_name_list(os.path.join(path, filename), fileType))
        elif os.path.isfile(os.path.join(path, filename)):
            if filename.endswith(fileType):
                file_list.append(filename.split('.')[0])
    return file_list

def text_alignement(emg_file_name_list, project_path, mfa_dataset_path):
    file_name_list = get_file_name_list(mfa_dataset_path, 'TextGrid')
    file_list = get_file_list(mfa_dataset_path, 'TextGrid')
    assert len(file_list) == len(file_name_list)
    base_directory = project_path
    for index, file_name in enumerate(file_name_list):
        if file_name in emg_file_name_list:
            # Split the filename to extract the parts
            parts = file_name.split('-')
            
            # Create the directory name and path
            dir_name = f"{parts[0]}-{parts[1]}"  # e.g., '5338-284437'
            new_dir = os.path.join(base_directory, dir_name)
            
            # Create the directory if it does not exist
            os.makedirs(new_dir, exist_ok=True)

            # Extract the number from the filename (e.g., '0003' from '5338-284437-0003.TextGrid')
            audio_number = int(parts[-1].split('.')[0])  
            
            
            # Construct the new file name
            new_file_name = f"{parts[0]}-{parts[1]}_{audio_number}_audio.TextGrid"  # e.g., '5338-284437_3_audio.TextGrid'
            destination_path = os.path.join(new_dir, new_file_name)

            # Copy the file to the new location with a new name
            print(f"Copying: {file_list[index]} to {destination_path}")
            
            # Use shutil.copy() to copy the file (specify the correct source path)
            # Uncomment the next line to actually copy the files
            shutil.copy(file_list[index], destination_path)



def other_emg_data(emg_file_name_list, train_emg_dataset_path, dataset_path, file_format, save_name):
    # Get the list of file names and their full paths
    file_name_list = get_file_name_list(dataset_path, file_format)
    file_list = get_file_list(dataset_path, file_format)

    # Check if the lengths of the two lists are equal
    if not len(file_name_list) == len(file_list):
        print(f'{file_name_list[:2]=}')
        print(f'{file_list[:2]=}')
        assert len(file_name_list) == len(file_list)

    base_directory = train_emg_dataset_path

    for index, file_name in enumerate(file_name_list):
        if file_name in emg_file_name_list:
            # Split the filename to extract the parts
            parts = file_name.split('-')
            if len(parts) != 3:
                continue

            # Create the directory name and path
            dir_name = f"{parts[0]}-{parts[1]}"  # e.g., '5338-284437'
            new_dir = os.path.join(base_directory, dir_name)

            # Create the directory if it does not exist
            os.makedirs(new_dir, exist_ok=True)

            # Extract the number from the filename (e.g., '0003' from '5338-284437-0003.TextGrid')
            audio_number = int(parts[-1].split('.')[0])  # Get '0003' and remove the extension

            # Construct the new file name
            new_file_name = f"{audio_number}_{save_name}.{file_format}"  # e.g., '5338-284437_3_audio.TextGrid'
            destination_path = os.path.join(new_dir, new_file_name)

            # Use shutil.copy() to copy the file (specify the correct source path)
            split_file_list = (file_list[index].rsplit('/', 1)[-1]).split('.')[0]
            if not split_file_list == file_name:
                print(f'{file_list[index]=}')
                print(f'{split_file_list=}')
                print(f'{file_name=}')
            assert split_file_list == file_name
            try:
                shutil.copy(file_list[index], destination_path)
                # Copy the file to the new location with a new name
                print(f"Copying: {file_list[index]} to {destination_path}")
            except PermissionError:
                print(f"PermissionError: Unable to copy {file_list[index]} to {destination_path}. Check permissions.")
            except Exception as e:
                print(f"An error occurred while copying {file_list[index]}: {e}")


def emg_data(train_emg_dataset_path, dataset_path, file_format, save_name):

    new_freq, old_freq = 1000, 800

    def upsample(signal, new_freq, old_freq):
        # Calculate the time points for the original and new signals
        times = np.arange(len(signal)) / old_freq
        upsampled_times = np.arange(0, times[-1] + 1 / new_freq, 1 / new_freq)
        
        # Perform linear interpolation for each channel
        if signal.ndim == 1:  # Single channel
            result = np.interp(upsampled_times, times, signal)
        else:  # Multi-channel
            result = np.array([np.interp(upsampled_times, times, channel) for channel in signal.T]).T

        return result

    def reverse_processing(raw_emg):
        # Step 1: Reverse the tanh using artanh
        # Note: Ensure that the values are within the valid range for artanh.
        # The valid range for artanh is (-1, 1), so clip the values accordingly.
        clipped_emg = np.clip(raw_emg, -1 + 1e-10, 1 - 1e-10)  # Clip to avoid singularities
        emg_after_tanh_reverse = np.arctanh(clipped_emg)

        # Step 2: Reverse the downscaling by multiplying by 100.0
        restored_emg = emg_after_tanh_reverse * 100.0
        return restored_emg

    assert file_format =='npy'
    # Get the list of file names and their full paths
    file_name_list = get_file_name_list(dataset_path, file_format)
    file_list = get_file_list(dataset_path, file_format)

    # Check if the lengths of the two lists are equal
    if not len(file_name_list) == len(file_list):
        print(f'{file_name_list[:2]=}')
        print(f'{file_list[:2]=}')
        assert len(file_name_list) == len(file_list)

    base_directory = train_emg_dataset_path

    for index, file_name in enumerate(file_name_list):
        # Split the filename to extract the parts
        parts = file_name.split('-')
        if len(parts) != 3:
            continue

        # Create the directory name and path
        dir_name = f"{parts[0]}-{parts[1]}"  # e.g., '5338-284437'
        new_dir = os.path.join(base_directory, dir_name)

        # Create the directory if it does not exist
        os.makedirs(new_dir, exist_ok=True)

        # Extract the number from the filename (e.g., '0003' from '5338-284437-0003.TextGrid')
        audio_number = int(parts[-1].split('.')[0])  # Get '0003' and remove the extension

        # Construct the new file name
        new_file_name = f"{audio_number}_{save_name}.{file_format}"  # e.g., '5338-284437_3_audio.TextGrid'
        destination_path = os.path.join(new_dir, new_file_name)


        # upsample & other reverse process
        split_file_list = (file_list[index].rsplit('/', 1)[-1]).split('.')[0]
        assert split_file_list == file_name
        fake = np.load(file_list[index])
        rever_fake = reverse_processing(upsample(fake, new_freq, old_freq))
        np.save(destination_path, rever_fake)
        print(f"Converted and save : {file_list[index]} to {destination_path}")
    return file_name_list

def get_file_list_jason(path, fileType):
    file_list = []
    for filename in os.listdir(path):
        # 如果是文件夹，则递归处理
        if os.path.isdir(os.path.join(path, filename)):
            file_list.extend(get_file_list_jason(os.path.join(path, filename), fileType))
        elif os.path.isfile(os.path.join(path, filename)):
            if filename.endswith(fileType):
                file_list.append(os.path.abspath(os.path.join(path, filename)).rsplit('_',1)[0])
    return file_list

def get_file_name_list_jason(path, fileType):
    file_list = []
    for filename in os.listdir(path):
        # 如果是文件夹，则递归处理
        if os.path.isdir(os.path.join(path, filename)):
            file_list.extend(get_file_name_list_jason(os.path.join(path, filename), fileType))
        elif os.path.isfile(os.path.join(path, filename)):
            if filename.endswith(fileType):
                file_list.append(filename.split('_')[0])
    return file_list

if __name__ == '__main__':
    project_path = 'path/to/project/v-ets/CoM2S_mix_scratch'

    sound_dataset_path = './raw_data_sheldan/LibriSpeech/dev-clean'
    mfa_dataset_path = './raw_data_sheldan/MFA_output'
    emg_dataset_path = './raw_data_sheldan/emg_generated'

    emg_data_format, audio_data_format, text_data_format = 'npy', 'flac', 'txt'


    run_emg_data, init_emg_data_path = True, False
    if run_emg_data:
        train_emg_dataset_path = project_path+'emg_data_all'
        print(f'{train_emg_dataset_path=}')
        if init_emg_data_path:
            print('remove')
            shutil.rmtree(train_emg_dataset_path)
        os.makedirs(train_emg_dataset_path)
        train_emg_dataset_path = project_path+'emg_data_all/'+'voiced_parallel_data'
        print(f'{train_emg_dataset_path=}')
        os.makedirs(train_emg_dataset_path, exist_ok=True)
        emg_file_name_list = emg_data(train_emg_dataset_path, emg_dataset_path, emg_data_format, 'emg')
        other_emg_data(emg_file_name_list, train_emg_dataset_path, sound_dataset_path, audio_data_format, 'audio')
        other_emg_data(emg_file_name_list, train_emg_dataset_path, sound_dataset_path, text_data_format, 'text')    


    '''text_alignements'''
    run_text_alignements, init_mfa_data_path = True, False
    if run_text_alignements:
        MFA_dataset_path = project_path+'text_alignments_all'
        if init_mfa_data_path:
            shutil.rmtree(MFA_dataset_path)
        os.makedirs(MFA_dataset_path, exist_ok=True)
        text_alignement(emg_file_name_list, MFA_dataset_path, mfa_dataset_path)

