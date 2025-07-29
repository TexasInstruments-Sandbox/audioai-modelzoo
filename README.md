# AudioAI-ModelZoo

This repository provides a collection of example Deep Neural Network (DNN) Models for various Audio tasks.

In order to run Deep Neural Networks on embedded hardware, they need to be optimized and converted into embedded friendly formats. We have converted/exported several models from the original training frameworks in PyTorch, Tensorflow and MxNet into these embedded friendly formats and is being hosted in this repository. In this process we also make sure that these models provide optimized inference speed on our SoCs, so sometimes minor modifications are made to the models wherever necessary. These models provide a good starting point for our customers to explore high performance Deep Learning on our SoCs.

**Notice**: The models in this repository are being made available for experimentation and development  - they are not meant for deployment in production.

## Supporting TI EdgeAI Processors and SDK Version

- Supporting Processors: AM62A, (TDA4VM, AM67A, AM68A, AM69A to be added)
- TIDL Version: 10_01_00_02


## Quick Start

### Download Models and Model Artifacts

Use the interactive model downloader to fetch pre-trained models:

```bash
cd models
./download_models.sh
```

The script provides an interactive menu to:
- View all available models on the server
- Select/deselect models for download
- Download selected models maintaining folder structure

Similarly, use the interactive model artifact downloader: 

```bash
cd models_artifacts
./download_artifacts.sh
```

## Pre-Trained Models

Pretrained models are located in the **[models](models)** folder. Following are the broad categories of models included. 


### Sound Classification (Audio-to-Class)

#### VGGish11

_**Inference in Jupyter Notebook**_: [inference/vggish11_sc/vggish_inference.ipynb](inference/vggish11_sc/vggish_inference.ipynb)


Start the Docker container:
```bash
cd ~/audioai-modelzoo/docker
./docker_run.sh
```

Below should be run inside the Docker container.
```bash
cd ~/audioai-modelzoo/notebooks/vggish11_sc
jupyter-lab --ip=<target_ip_address> --no-browser --allow-root
jupyter notebook --ip=<target_ip_address> --no-browser --allow-root --port=8888
```

Open a browser on a remote PC to enter the URL displayed on the terminal for Jupyter lab.

Python script version: Below should be run inside the Docker container.
```bash
python3 ./vggish_infer_audio.py --audio-file sample_wav/139951-9-0-9.wav --detailed-report
```

### Speech Enhancement / Audio Denoising (Audio-to-Audio)

#### GTCRN

_**Inference in Jupyter Notebook**_: [inference/gtcrn_se/gtcrn_inference.ipynb](inference/gtcrn_se/gtcrn_inference.ipynb)


```bash
cd inference/gtcrn_se
python3 -m venv .venv
source .venv/bin/activate
pip install wheel setuptools pip --upgrade
pip install -r requirements.txt
```

In the venv inside the docker container:
```bash
jupyter notebook --ip=<target_ip_address> --no-browser --allow-root --port=8888
```

### Foundational Models

#### UNet
To be added
