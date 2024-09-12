<h1 align="center">Reliable Spatial-Temporal Voxels for Multi-Modal Test-Time Adaptation</h1>

<p align="center"><strong>
    <a href = "https://sites.google.com/view/haozhicao">Haozhi Cao</a><sup>1</sup>,
    <a href = "https://xuyu0010.wixsite.com/xuyu0010">Yuecong Xu</a><sup>2</sup>,
    <a href = "https://marsyang.site/">Jianfei Yang</a><sup>3*</sup>,
    <a href = "https://pamphlett.github.io/">Pengyu Yin</a><sup>1</sup>,
    <a href = "https://scholar.google.com/citations?user=qcLKoccAAAAJ&hl=en">Xingyu Ji</a><sup>1</sup>,
    <a href = "https://scholar.google.com/citations?user=XcV_sesAAAAJ&hl=en">Shenghai Yuan</a><sup>1</sup>,
    <a href = "https://scholar.google.com.sg/citations?user=Fmrv3J8AAAAJ&hl=en">Lihua Xie</a><sup>1</sup>
</strong></p>

<p align="center"><strong>
    <a href = "https://www.ntu.edu.sg/cartin">1: Centre for Advanced Robotics Technology Innovation (CARTIN), Nanyang Technological University</a><br>
    <a href = "https://cde.nus.edu.sg/ece/">2: Department of Electrical and Computer Engineering, National University of Singapore</a><br>
    <a href = "https://www.ntu.edu.sg/mae">3: School of MAE</a> and <a href = "https://www.ntu.edu.sg/eee">School of EEE</a>, Nanyang Technological University<br>
</strong></p>

<p align="center"><strong> 
    <a href = "https://arxiv.org/abs/2403.06461">&#128196; [Arxiv]</a> | 
    <a href = "https://sites.google.com/view/eccv24-latte">&#128190; [Project Site]</a> |
    &#128214; [OpenAccess]
</strong></p>

## :scroll: About Latte (ECCV 2024)

**Latte** is a MM-TTA method that leverages esitmated 3D poses to retrieve reliable spatial-temporal voxels for Test-Time Adaptatipn (TTA). The overall structure is as follows.

<p align="middle">
  <img src="figs/Main_Method.jpg" width="600" />
</p>


## Installation and Prerequisite

To ease the effort during environment setup, we recommend you leverage [Docker](https://www.docker.com/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/ai-enterprise/deployment-guide-vmware/0.1.0/docker.html). With Docker installed, you can locally build the docker image for Latte using [this Dockerfile](Dockerfile) by running ```docker build -t mopa docker/ ```.

You can then run a container using the docker image. The next step is to install some additional prerequisites. To do so, go to this repo folder and run ```bash install.sh``` within the built container (you may ignore the warning saying some package versions are incompatible).


## Dataset Preparation
Please refer to [DATA_PREPARE.md](latte/data/DATA_PREPARE.md) for the data preparation and pre-processing details.


## :eyes: Updates
* [2024.09] Installation and data preparation details released. Full realse will be available soon!
* [2024.08] We are now refactoring our code and code will be available shortly. Stay tunned!
* [2024.07] Our paper is accepted by ECCV 2024! Check our paper on arxiv [here](https://arxiv.org/abs/2403.06461).


## :writing_hand: TODO List

- [x] Initial release. :rocket:
- [x] Add installation and prerequisite details.
- [x] Add data preparation details.
- [ ] Add training details.
- [ ] Add evaluation details.


## :clap: Acknowledgement
We greatly appreciate the contributions of the following public repos:
- [torchsparse](https://github.com/mit-han-lab/torchsparse)
- [SPVNAS](https://github.com/mit-han-lab/spvnas)
- [SalsaNext](https://github.com/TiagoCortinhal/SalsaNext)
- [KISS-ICP](https://github.com/PRBonn/kiss-icp)
- [xMUDA](https://github.com/valeoai/xmuda)

## :envelope: Contact
For any further questions, please contact Haozhi Cao (haozhi002@ntu.edu.sg)

## :pencil: Citation
```
@article{cao2024reliable,
  title={Reliable Spatial-Temporal Voxels For Multi-Modal Test-Time Adaptation},
  author={Cao, Haozhi and Xu, Yuecong and Yang, Jianfei and Yin, Pengyu and Ji, Xingyu and Yuan, Shenghai and Xie, Lihua},
  journal={arXiv preprint arXiv:2403.06461},
  year={2024}
}
```