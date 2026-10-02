# Docker-based Setup (Alternative)

The recommended approach is the native Python venv described in the [README](../README.md). This document describes the alternative Docker-based flow on the AM62A.

## Prerequisites

```bash
cd ~/tidl/audioai-modelzoo
./download_models.sh -y
./download_artifacts.sh -y
```

## Docker Image Setup

This repository uses a two-stage Docker build process (see the [docker](../docker) folder). The base image contains all dependencies and is pre-built and available from GitHub Container Registry. The TI-specific image adds processor-specific libraries on top of the base.

Pull the pre-built base image and build the TI image:

```bash
docker pull ghcr.io/texasinstruments-sandbox/audioai-base:11.1.0
docker tag ghcr.io/texasinstruments-sandbox/audioai-base:11.1.0 audioai-base:11.1.0
cd docker
./docker_build_ti.sh
```

If you want to build the base image from scratch instead of pulling it, run `./docker_build_base.sh` before building the TI image.

## Start Jupyter Server

Launch the container:

```bash
~/tidl/audioai-modelzoo/docker/docker_run.sh
```

Inside the container, start Jupyter Lab:

```bash
./jupyter_lab.sh
```

The script displays a highlighted access URL. Open it in your browser to access Jupyter Lab with three inference notebooks pre-loaded in tabs.

![JupyterLab running inside the Docker container](jupyter_lab_screenshot.jpg)

## Python Scripts

Run the inference scripts inside the container:

```bash
cd ~/tidl/audioai-modelzoo/inference/vggish11_sc
python3 vggish_infer_audio.py --audio-file sample_wav/139951-9-0-9.wav

cd ~/tidl/audioai-modelzoo/inference/yamnet_sc
python3 yamnet_infer_audio.py --audio-file samples/miaow_16k.wav
```
