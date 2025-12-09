# AudioAI-ModelZoo

A collection of optimized Deep Neural Network (DNN) models for Audio Tasks on TI EdgeAI processors. Models are converted from PyTorch and TensorFlow into embedded-friendly formats optimized for TI SoCs.


**Notice**: The models in this repository are being made available for experimentation and development - they are not meant for deployment in production.

## System Requirements

- **Processors**: AM62A (extensible to TDA4VM, AM67A, AM68A, AM69A)
- **TIDL Version**: 11_01_06_00

## Quick Start

### Git pull the project

On the Linux command line on the target (AM62A)

```bash
mkdir ~/tidl && cd ~/tidl
git clone https://github.com/TexasInstruments-Sandbox/audioai-modelzoo.git
cd audioai-modelzoo
```

### Download Models and Model Artifacts

```bash
./download_models.sh -y
./download_artifacts.sh -y
```

Both scripts provide interactive menus to select and download models.

### Build Docker Image

```bash
cd docker
./docker_build.sh
```

## Start Jupyter Server

Launch the Docker container:

```bash
/root/tidl/audioai-modelzoo/docker/docker_run.sh
```

Inside the Docker container, start Jupyter Lab:

```bash
cd ~/tidl/audioai-modelzoo/inference
jupyter-lab --ip=$TARGET_IP --no-browser --allow-root
```

Access Jupyter Lab from your browser using the URL displayed in the terminal. 

## Pre-Trained Models

Models are located in the **[models](models)** folder.

### Sound Classification (Audio-to-Class)

#### VGGish11

_**Inference in Jupyter Notebook**_: [inference/vggish11_sc/vggish_inference.ipynb](inference/vggish11_sc/vggish_inference.ipynb)


Python script version: Below should be run inside the Docker container.

```bash
python3 vggish_infer_audio.py --audio-file sample_wav/139951-9-0-9.wav --detailed-report
```

#### YAMNet

_**Inference in Jupyter Notebook**_: [inference/yamnet_sc/yamnet_inference.ipynb](inference/yamnet_sc/yamnet_inference.ipynb)

Python script version: Below should be run inside the Docker container.

```bash
python3 yamnet_infer_audio.py --audio-file samples/miaow_16k.wav --detailed-report
```

### Speech Enhancement (Audio-to-Audio)

#### GTCRN

_**Inference in Jupyter Notebook**_: [inference/gtcrn_se/gtcrn_inference.ipynb](inference/gtcrn_se/gtcrn_inference.ipynb)

