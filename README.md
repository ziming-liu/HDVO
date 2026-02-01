# HDVO: Hybrid Dense Direct Visual Odometry

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1+-orange.svg)](https://pytorch.org/)

</div>

<div align="center">
  <img src="docs/imgs/inria.png" height="80"/>
</div>

<br>

Official PyTorch implementation of **HDVO** from [ACENTAURI team @ INRIA](https://team.inria.fr/acentauri/).

---

## 🔥 News

- **[2024-02]** Initial code release

## ✨ Highlights

- **Hybrid Architecture**: Combines data-driven deep learning with model-based geometric constraints for robust visual odometry
- **Unified Framework**: Joint stereo depth estimation and visual odometry in a single framework
- **State-of-the-art Performance**: Achieves competitive results on KITTI Odometry and Virtual KITTI 2 benchmarks
- **Flexible Design**: Modular architecture allows easy integration and customization

## 📋 Table of Contents

- [Installation](#-installation)
- [Dataset Preparation](#-dataset-preparation)
- [Model Zoo](#-model-zoo)
- [Getting Started](#-getting-started)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)
- [Contact](#-contact)

## 🛠️ Installation

### Prerequisites

- Python == 3.8
- PyTorch == 2.1.0a0 (Jetson version)
- JetPack == 5.1.2
- NVIDIA Jetson Orion device

### Environment Setup

**Note:** This branch is specifically for NVIDIA Jetson Orion devices. For standard GPU installation, please switch to the main branch.

```bash
# Clone the repository
git clone https://github.com/ziming-liu/hdvo.git
cd hdvo

# Create conda environment
conda create -n hdvo python==3.8
conda activate hdvo

# Install PyTorch Jetson version
# Refer to: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html
# v512 is the JetPack version (5.1.2)
export TORCH_INSTALL=https://developer.download.nvidia.cn/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl

python3 -m pip install --upgrade pip
python3 -m pip install numpy==1.26.1
python3 -m pip install --no-cache $TORCH_INSTALL

# Install mmcv-full 1.6.0
cd mmcv-full-1.6.0 && python setup.py develop
cd ..

# Install other dependencies
source $(conda info --base)/etc/profile.d/conda.sh && conda activate hdvo && cat requirements_orion.txt | while read package; do    if [ -n "$package" ] && [[ ! "$package" =~ ^#.* ]]; then      echo "Installing: $package";     pip install "$package" || echo "Skip failed packages: $package";   fi; done

# Install torchvision from source
git clone https://github.com/pytorch/vision.git
cd vision
git checkout tags/v0.19.0
pip install -e .
cd ..

# Install the package hdvo
pip install -e .

# Install openrox
bash build_openrox.sh $OPENROX_DIR $HDVO_DIR

# or use compiled openrox file
$HDVO_DIR=/your_path
export LD_LIBRARY_PATH=$HDVO_DIR/openrox/cmake:$LD_LIBRARY_PATH

# rox_odometry_module.so has been included in this repo
```

## 📦 Dataset Preparation

We support the following datasets:

- **KITTI Odometry**: For visual odometry training and evaluation
- **Virtual KITTI 2**: For training and testing

Annotations for `KITTI Odometry` has already existed under `annotations/kittiodometry`.

Please refer to [prepare_dataset.md](docs/prepare_dataset.md) for detailed instructions on downloading and organizing datasets.

## 🎯 Model Zoo

### Stereo Depth Estimation

| Model | Dataset | Config | Checkpoint |
|-------|---------|--------|------------|
| HDVO | KITTI Odometry | [config](configs/hdvo) | [ckp](pretrained/iter_40000.pth) |
| HDVO | Virtual KITTI 2 | Coming soon | Coming soon
<!-- ### Quick Demo

```python
# Coming soon
``` -->

### Inference on Jetson

For faster inference on NVIDIA Jetson devices, use `test_on_jetson.py` with optimization options:

**Basic inference:**
```bash
python tools/test_on_jetson.py \
  configs/hdvo/stereohdvo_posesup_coex_kittiodom_huberloss.py \
  pretrained/iter_40000.pth \
  --test_seq_id 09
```

**FP16 inference (faster with lower memory):**
```bash
python tools/test_on_jetson.py \
  configs/hdvo/stereohdvo_posesup_coex_kittiodom_huberloss.py \
  pretrained/iter_40000.pth \
  --test_seq_id 09 \
  --fp16
```

**FP16 + JIT optimization (maximum speed):**
```bash
python tools/test_on_jetson.py \
  configs/hdvo/stereohdvo_posesup_coex_kittiodom_huberloss.py \
  pretrained/iter_40000.pth \
  --test_seq_id 09 \
  --fp16 \
  --compile
```

**Note:** FP16 and JIT optimization can provide 1.5-2x speedup on Jetson platforms with minimal accuracy loss.



## 📄 License

This project is released under the [Apache 2.0 license](LICENSE).

## 🙏 Acknowledgements

This codebase is built upon several excellent open-source projects:

- [MMAction](https://github.com/open-mmlab/mmaction) - Framework structure and utilities
- [OpenRox](https://github.com/robocortex/openrox) - Computer vision algorithms

We thank the authors for their great work and open-source contributions.

## 📧 Contact

For questions and discussions, please contact:

- **Ziming Liu**: liuziming.email@gmail.com

You can also open an issue in this repository for bug reports and feature requests.

---

<div align="center">
Made with ❤️ by ACENTAURI team @ INRIA
</div> 