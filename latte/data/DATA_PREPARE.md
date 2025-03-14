# Dataset Preparation
Below we provide how to prepare the dataset for MoPA, including:
* [A2D2](#1-a2d2)
* [NuScenes](#2-nuscenes)
* [SemanticKITTI](#3-semantickitti)
* [Synthia](#4-synthia)
* [Poses Preparation](#5-poses-preparation)

Before getting started, we recommand you to create a folder  ```latte/datasets``` to store the soft links to each datasets for better data arrangement.

## 1. A2D2
Our preprocessing procedures of A2D2 are identical to [xMUDA](https://github.com/valeoai/xmuda) and [MoPA](https://github.com/AronCao49/MoPA). First download the raw A2D2 dataset from [their official website](https://a2d2.audi/a2d2/en.html). Then, create a soft link ```ln -sfn /path/to/raw/a2d2 latte/datasets/a2d2```, and subsequently run the following preprocessing command:
```bash
$ python latte/data/a2d2/preprocess.py
```

## 2. NuScenes
Similar to A2D2, our preprocessing on NuScenes is the same as [xMUDA](https://github.com/valeoai/xmuda) and [MoPA](https://github.com/AronCao49/MoPA). After download the NuScenes dataset from their [website](https://www.nuscenes.org/nuscenes), create a soft link ```ln -sfn /path/to/raw/nuscenes latte/datasets/nuscenes```, and then run the following command:
```bash
$ python latte/data/nuscenes/preprocess.py
```

## 3. SemanticKITTI
We employ a scan-by-scan loading strategy in our repo as in [MoPA](https://github.com/AronCao49/MoPA). After following the downloading instruction of [xMUDA](https://github.com/valeoai/xmuda), create a soft link ```ln -sfn /path/to/raw/semantickitti latte/datasets/semantic_kitti```, and then run the following command:
```bash
$ python latte/data/semantic_kitti/preprocess.py
```

## 4. Synthia
The preprocessing procedure of Synthia is mainly about simulating LiDAR point clouds given the dense depth input as in [CoMAC](https://sites.google.com/view/mmcotta). To prepare the Synthia dataset for Latte, first download the SYNTHIA-RAND-CITYSCAPES from [the dataset homepage](https://synthia-dataset.net/downloads/). Then, decompress the downaloaded file and create a soft link ```ln -sfn /path/to/raw/synthia latte/datasets/synthia```, resulting in the file organization as below:

📦synthia_link <br> 
┣ 📂RAND_CITYSCAPES <br>
┃ ┣ 📂Depth  <br>
┃ ┣ 📂RGB  <br>
┃ ┣ 📂GT  <br>
┗ ┗...

You can subsequently run the following command to conduct the preprocessing we need. After the preprocessing done, you will find the additional ```bin```, ```Lidar```, and ```Label``` directories under the folder ```RAND_CITYSCAPES```:
```bash
$ python latte/data/synthia/preprocess.py
```

## 5. Poses Preparation
Poses are estimated by the default setting of [KISS-ICP](https://github.com/PRBonn/kiss-icp). We provide the extracted poses in [google drive](https://drive.google.com/drive/folders/1SkeOBXjtGXZzHDrKC9s_Cdh_N6SS_tdB?usp=sharing). To enalbe the TTA process of Latte, first download the respective poses zip file (e.g., poses.zip for SemanticKITTI), and then decompress the zip file under the link to each dataset (e.g., ```latte/datasets/semantic_kitti/```, and ```latte/datasets/nuscenes/```).

Also, if you prefer generating poses by yourself, you can just leverage our forked version of [KISS-ICP](https://github.com/PRBonn/kiss-icp) (which is modified for more suitable I/O for this work). To do so, first install KISS-ICP using the following command:
```bash
$ git clone git+https://github.com/AronCao49/kiss-icp.git && cd kiss-icp && make editable
```
Then you can use the ```preproces.py``` for each datasets to generate the estimated poses.


