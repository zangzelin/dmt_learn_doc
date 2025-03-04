# Configuring Python Environment


To install the corresponding environment, you can follow the steps below.


## Step 1: Create and Activate Conda Environment

```bash
conda create -n dmtlearn python=3.10  
conda activate dmtlearn

```

## Step 2: Install PyTorch with CUDA Support
Install PyTorch, TorchVision, and TorchAudio with CUDA 12.3 support for GPU acceleration.

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.3 -c pytorch -c nvidia
```

## Step 3: Additional Dependencies
There are other dependencies specified in the install_env.sh script, you can install them using the following command. Make sure that `install_env.sh` is available in your directory.
```bash
bash install_env.sh
