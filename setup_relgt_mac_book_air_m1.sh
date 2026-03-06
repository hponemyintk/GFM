#!/bin/bash

python -m venv relgt_env
source relgt_env/bin/activate

pip install --upgrade pip wheel setuptools

pip install torch torchvision torchaudio

pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
-f https://data.pyg.org/whl/torch-2.5.0+cpu.html

pip install torch-geometric

pip install \
wandb absl-py tensorboard einops matplotlib progressbar2 pandas \
numba networkx scikit-network ipykernel tqdm

pip install \
kmeans-pytorch torchviz fastcluster opentsne ogb kmedoids \
relbench pytorch_frame[full] sentence-transformers h5py pynvml

git clone https://github.com/snap-stanford/relgt.git
