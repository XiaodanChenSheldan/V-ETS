# Confidence-Based Self-Training for EMG-to-Speech: Leveraging Synthetic EMG for Robust Modeling

This repository contains the implementation and resources for the V-ETS project presented in the paper:

**Confidence-Based Self-Training for EMG-to-Speech: Leveraging Synthetic EMG for Robust Modeling**  
*Authors: Xiaodan Chen, Xiaoxue Gao, Mathias Quoy, Alexandre Pitti, Nancy F. Chen*  
[ASRU 2024](http://dx.doi.org/10.48550/arXiv.2506.11862) 

This work builds on the following repositories:  
- [Morrison et al. (2021). Chunked Autoregressive GAN for Conditional Waveform Synthesis](https://doi.org/10.48550/ARXIV.2110.10139) | [GitHub](https://github.com/descriptinc/cargan)
- [van Niekerk et al. (2022). A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion. ICASSP 2022. IEEE](https://doi.org/10.1109/icassp43922.2022.9746484) | [GitHub](https://github.com/bshall/soft-vc)
- [Gaddy & Klein (2020). Digital Voicing of Silent Speech](https://doi.org/10.48550/ARXIV.2010.02960) | [GitHub](https://github.com/dgaddy/silent_speech)
- [Scheck & Schultz (2023). STE-GAN: Speech-to-Electromyography Signal Conversion Using GANs. INTERSPEECH 2023. ISCA](https://doi.org/10.21437/interspeech.2023-174) | [GitHub](https://github.com/scheck-k/ste-gan/)

---
## Open access Libri-EMG dataset
In support of future research, we provide the proposed [Libri-EMG](https://zenodo.org/records/16788832) dataset—an open-access, time-aligned, multi-speaker voiced EMG and speech recordings collection. This dataset contains **8.3 hours** of EMG-speech data covering a diverse set of **1,532 speakers** across both male and female categories.

The speech data used in this project is derived from [LibriSpeech](https://doi.org/10.1109/icassp.2015.7178964)

[Audio Examples and Comparisons](https://xiaodanchensheldan.github.io/v-ets/)



### Data Structure

The dataset is organized as follows:

- **Audio data:**  
  `v-ets/CoM2S_mix_scratch/emg_data_all/..._audio.flac`

- **EMG data:**  
  `v-ets/CoM2S_mix_scratch/emg_data_all/..._emg.npy`

- **Transcript data:**  
  `v-ets/CoM2S_mix_scratch/emg_data_all/..._text.txt`

- **Forced alignment data:**  
  `v-ets/CoM2S_mix_scratch/text_alignments_all/..._audio.TextGrid`

*Note: The ellipsis (`...`) represents specific speaker and utterance identifiers in the file paths.*


## Phoneme-Error-Based Filtered Synthetic Libri-EMG Data
0. **Installation**

    Set up the required packages by creating a Python 3.10 environment named ste-gan using Conda.
    ```bash
    cd path/to/project/synthetic_emg_gen
    conda create -n ste-gan python=3.10
    conda activate ste-gan
    pip install -r requirements.txt
    pip install -e .
    ```

    The example code in this repo uses the `dev-clean` dataset of LibriSpeech as a small demo for fast testing. Please download the `dev-clean` dataset from [LibriSpeech](https://www.openslr.org/12) and place it under `path/to/project/v-ets/synthetic_emg_gen/raw_data_sheldan`:
    ```bash
    raw_data_sheldan/
    ├── LibriSpeech/
    │   └── dev-clean/
    │           ├── 84/
    │           └── ...
    ├── alignment_processing.py
    ```
    

1. **Prepare phoneme classes:**

    **If you don't need to work with any dataset other than `dev-clean`, skip to the second step.**


    To maintain consistency with LibriSpeech, this project uses the `english_us_arpa.dict` and `english_us_arpa.zip` files to generate phoneme classes. If you want to use this for other sound datasets other than LibriSpeech, please refer to https://mfa-models.readthedocs.io/en/latest/index.html (or useful link: https://lingmethodshub.github.io/content/tools/mfa/mfa-tutorial#obtaining-acoustic-models) to install the virtual environment and download the corresponding dictionary and model:

    ```bash
    mfa model download acoustic english_us_arpa
    ```

    ```bash
    mfa model download dictionary english_us_arpa`
    ```

    Before running the phoneme alignment script, please update the following parameters in `alignment_processing.py`:

    - `AudioFileType (e.g., .flac)`
    - `sound_dataset_path` — path to your audio dataset  
    - `dictionary_path` — path to the phoneme dictionary files (`english_us_arpa.dict`)  
    - `acoustic_model_path` — path to your acoustic model  

    (To speeds up the process, especially for large datasets, --num_jobs tells it how many parallel processes to use for alignment, it can max out CPU usage.)
    

    Then run:
    ```bash
    cd path/to/project/synthetic_emg_gen/raw_data_sheldan
    python alignment_processing.py
    ```
    It will generate transcript correspondingly under each audio file and phoneme class under `path/to/project/v-ets/synthetic_emg_gen/raw_data_sheldan/MFA_output` 



2. **Prepare data for synthetic EMG generation:**

    ```bash
    cd path/to/project/v-ets/synthetic_emg_gen
    python -m scripts.prep_test_sheldan
    ```
    This will overwrite the test dataset under `ste-gan/data/sheldan_complete/test`.

    (if you get `AttributeError: module 'torch.nn.utils.parametrizations' has no attribute 'weight_norm'` error, symply go to `path/to/torch/.cache/torch/hub/bshall_hubert_main/hubert/model.py` and change: `self.conv = nn.utils.parametrizations.weight_norm(`to: `self.conv = nn.utils.weight_norm(` ).

    ```bash
    python -m ste_gan.test_sheldan_saveemg --filter_threshold 0.5
    ```

    Now, the high-quality time-aligned EMG data is available under `path/to/project/v-ets/synthetic_emg_gen/raw_data_sheldan/emg_generated`. If everything's right, we should have:
    ```bash
    raw_data_sheldan/
    ├── emg_generated/
    ├── LibriSpeech/
    ├── MFA_output/
    ├── alignment_processing.py
    ```


3. **OPTIONAL: Ensure that all datasets are placed in the specified directories with consistent file formats for the EMG encoder (transduction model):**


    Before running the phoneme alignment script, set the following parameters in `alignment_processing.py` to match your dataset structure and file formats:

    - `sound_dataset_path` - Path to audio dataset
    - `mfa_dataset_path` - Path to MFA-aligned phoneme data in step 1
    - `emg_dataset_path` - Path to generated synthetic EMG output in step 2

    - `audio_data_format` - audio file format (e.g., .flac for LibriSpeech)
    - `text_data_format` - Audio transcript format (e.g., .txt for LibriSpeech)  

    Then run:
    ```bash
    python get_emg_encoder_data.py
    ```
    
    If you set `init_data_path=True`, the entire ~8 hours dataset located at `path/to/project/CoM2S_mix_scratch/emg_data_all` and all forced alignment files in `path/to/project/CoM2S_mix_scratch/text_alignments_all` will be deleted.
    Only enable this option if you intend to regenerate your own EMG–speech paired data from scratch.

4. **Quit the virtual environment:**
    ```bash
    conda deactivate
    ```

## Train-from-Scratch Approach for Synthetic Data Investigation
0. **Installation**
    ```bash
    cd path/to/project/CoM2S_mix_scratch
    conda env create -f environment.yml
    conda activate v_ets
    ```
    Download [Libri-EMG](https://zenodo.org/records/16788832) data and eExtract the original dataset inside the `CoM2S_mix_scratch/` directory, so the folder structure looks like::
    ```bash
    CoM2S_mix_scratch/
    ├── emg_data_all/
    ```

    Download the pre-trained DeepSpeech 0.7.0 model files for evaluation to ensure consistency with the baseline model. Newer DeepSpeech versions (e.g., 0.9.3) can be used if compatible with 0.7.x models.
    ```bash
    curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.7.0/deepspeech-0.7.0-models.pbmm
    curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.7.0/deepspeech-0.7.0-models.scorer
    ```
    
1. **Train-from-scratch with optimal 1:1 real-synthetic ratio**
    ```bash
    python -m transduction_model_sheldan_mix_scratch --hifigan_checkpoint hifigan_finetuned/checkpoint
    ```

2. **Evaluation**

    To evaluate on baseline data:
    ```bash
    python -m evaluate_baseline_nosilent --models output_sheldan_mix_scratch_voiced/model.pt --hifigan_checkpoint hifigan_finetuned/checkpoint --output_directory evaluation_baselinenosilent_scratch_voiced_baseline
    ```
    Audio samples from the baseline test dataset are located in the `evaluation_baselinenosilent_scratch_voiced_baseline/` directory.

    To run test on synthetic data:
    ```bash
    python -m evaluate --models output_sheldan_mix_scratch_voiced/model.pt --hifigan_checkpoint hifigan_finetuned/checkpoint --output_directory evaluation_scratch_voiced
    ```
    Audio samples from the synthetic test dataset are located in the `evaluation_baselinenosilent_scratch_voiced_baseline/` directory.

3. **Quit the virtual environment:**
    ```bash
    conda deactivate
    ```



