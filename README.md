# Density-Based Flow Mask Integration via Deformable Convolution for Video People Flux Estimation

Chang-Lin Wan, Feng-Kai Huang, Hong-Han Shuai

This is the official implementation of [Density-Based Flow Mask Integration via Deformable Convolution for Video People Flux Estimation](https://openaccess.thecvf.com/content/WACV2024/papers/Wan_Density-Based_Flow_Mask_Integration_via_Deformable_Convolution_for_Video_People_WACV_2024_paper.pdf) (accepted by WACV 2024)

![这是图片](/src/figures/framework.png "Model Architecture")


## Requirements
- Python 3.8
- Install dependencies:
  ```sh
  pip3 install -r requirements.txt
  ```


## Datasets
- Our CARLA dataset can be downoaded from [here](https://drive.google.com/file/d/1hycxlqE66QGXsOo-HMWyi3WUlF2TII8r/view?usp=drive_link).
- HT21 dataset can be downloaded from [here](https://motchallenge.net/data/Head_Tracking_21/)
- Folder structure:
  ```
  ./src
  ./dataset
  └── CARLA
      ├── test
      ├── train
      ├── train.txt
      ├── test.txt
      └── val.txt
  ```

## Training
- For colorization pretrian, please modify the `config.py` `__C.task = "LAB"`, you can see more details in `config.py`
- After colorization, change `__C.task = "DEN"` and run
  ```sh
  python train.py
  ```
  `__C.DATASET` can change the training dataset.

## Test
- The pretrained weight for HT21 can download from [here](https://drive.google.com/file/d/1G_J5HKOjEJn5LtTAHue9Exo7qMBiRv3O/view?usp=sharing)-
- For CARLA dataset:
  ```sh
  python test_CARLA.py
  ```
- For HT21 dataset:
  ```sh
  python test_HT21.py
  ```

 

## Citation
If you find our work is relevant to your research, please cite:
```
@inproceedings{wan2024density,
  title={Density-Based Flow Mask Integration via Deformable Convolution for Video People Flux Estimation},
  author={Wan, Chang-Lin and Huang, Feng-Kai and Shuai, Hong-Han},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={6573--6582},
  year={2024}
}
```
