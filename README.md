# DPIR
**Differentiable Point-based Inverse Rendering**

[Paper](https://arxiv.org/abs/2312.02480) 

Hoon-Gyu Chung, Seokjun Choi, Seung-Hwan Baek

CVPR, 2024

## Abstract
We present differentiable point-based inverse rendering, DPIR, an analysis-by-synthesis method that processes images captured under diverse illuminations to estimate shape and spatially-varying BRDF. To this end, we adopt point-based rendering, eliminating the need for multiple samplings per ray, typical of volumetric rendering, thus significantly enhancing the speed of inverse rendering. To realize this idea, we devise a hybrid point-volumetric representation for geometry and a regularized basis-BRDF representation for reflectance. The hybrid geometric representation enables fast rendering through point-based splatting while retaining the geometric details and stability inherent to SDF-based representations. The regularized basis-BRDF mitigates the ill-posedness of inverse rendering stemming from limited light-view angular samples. We also propose an efficient shadow detection method using point-based shadow map rendering. Our extensive evaluations demonstrate that DPIR outperforms prior works in terms of reconstruction accuracy, computational efficiency, and memory footprint. Furthermore, our explicit point-based representation and rendering enables intuitive geometry and reflectance editing. The code will be publicly available.

<p align="center">
    <img src='docs/intro.png' width="800">
</p>

## Installation
We recommend you to use Conda environment. Install pytorch3d following [INSTALL.md](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).

```bash
conda create -n pytorch3d python=3.9
conda activate pytorch3d
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install numpy matplotlib tqdm imageio
pip install scikit-image matplotlib imageio plotly opencv-python
```

## Dataset
We utilized multi-view multi-light image dataset([DiLiGenT-MV Dataset](https://sites.google.com/site/photometricstereodata/mv?authuser=0)) and photometric image dataset.

Multi-view multi-light image dataset was preprocessed following [PS-NeRF](https://github.com/ywq/psnerf), which contains 5 objects.

Photometric image dataset was rendered by Blender, which contains 4 objects.

You can download dataset from Google Drive([]) and put them in the corresponding `data/` folder.

## Model
You can download pre-trained models for each dataset from Google Drive([]) and put them in the corresponding `output/` folder,

## Train
You can train multi-view multi-light dataset(DiLiGenT-MV Dataset) or photometric dataset.
If you want to train DiLiGenT-MV Dataset,
```bash
cd 
python main.py --datadir xxx --dataname hotdog --basedir xxx --data_r 0.012 --splatting_r 0.015
```
## Evaluation


