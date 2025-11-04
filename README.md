<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center">GSPOSE: Textureless Object Pose Estimation Guided by Gaussian Splatting</h1>
  <div align="center"></div>
</p>


## Environments

You can download the submodules at https://pan.baidu.com/s/1uwcK5Ntrk5c2to95Cean_Q. The key is 8mg8.

We build the Python environment using [Anaconda](https://www.anaconda.com/download/):
```shell
cd GS2POSE
git submodule update --init --recursive
conda create -n gs2pose python=3.9 
conda activate gs2pose

pip install -r requirements.txt
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```


## Datasets
We used two datasets for training and evaluation.

### LineMod

Download the preprocessed LineMOD dataset from (https://drive.google.com/drive/folders/19ivHpaKm9dOrr12fzC8IDFczWRPFxho7) refer from DenseFusion.

### LineMod-Occlusion
Download the LMO dataset all test images from (https://bop.felk.cvut.cz/datasets/)

The train data structure is as follows:

```
./coarse_net 
├── 000001
│    ├── train
│    │    └── rgb
│    │    └── render
└── 000002
     └── ...
```

The test data structure is as follows:

```
./data 
├── lm
│    ├── 000001
│    │    └── scene_gt.json
│    │    └── mask_visib
│    │    └── depth
│    │    └── rgb
│    ├── 000002
│    │    └── ...
└── lmo
     └── ...
```

## Reconstruction the ply model

```
python reconstruction.py
```

The ply model can be found in output/{index}/point_cloud/iteration_3600/point_cloud.ply

## Train

```
cd coarse_net
python coarse_train.py
```

The checkpoint can be found at ./coarse_net/checkpoints

## Coarse Estimation

```
# Get the nocs image
cd coarse_net
python coarse_test.py
```

The nocs images can be found at ./result/{index}

## Refine Estimation

```
cd ..
python pnp_xh.py
```
