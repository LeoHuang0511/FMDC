# Density-Based Flow Mask Integration via Deformable Convolution for Video People Flux Estimation

Chang-Lin Wan, Feng-Kai Huang, Hong-Han Shuai

This is the official implementation of [Density-Based Flow Mask Integration via Deformable Convolution for Video People Flux Estimation](https://openaccess.thecvf.com/content/WACV2024/papers/Wan_Density-Based_Flow_Mask_Integration_via_Deformable_Convolution_for_Video_People_WACV_2024_paper.pdf).

![这是图片](/src/figures/framework.png "Model Architecture")


## Requirements
- Python 3.8
- Install dependencies:
  ```sh
  pip3 install -r requirements.txt
  ```


## Datasets
- Our CARLA dataset can be downoaded from [here](https://nycu1-my.sharepoint.com/:u:/g/personal/s311505011_ee11_m365_nycu_edu_tw/ESv4FNy51fJEiCfByfgOea0B5yxE4JZh4JmTTGtEGTdLQw?e=UlorCN).
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
- The pretrained weight for HT21 can download from [here](https://nycu1-my.sharepoint.com/personal/s311505011_ee11_m365_nycu_edu_tw/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fs311505011%5Fee11%5Fm365%5Fnycu%5Fedu%5Ftw%2FDocuments%2FWACV24%5FPeople%5FFlow%2Fep%5F13%5Fiter%5F33000%5Fmae%5F13%2E118%5Fmse%5F13%2E494%5Fseq%5FMAE%5F0%2E237%5FWRAE%5F0%2E273%5FMIAE%5F2%2E962%5FMOAE%5F1%2E968%2Epth&parent=%2Fpersonal%2Fs311505011%5Fee11%5Fm365%5Fnycu%5Fedu%5Ftw%2FDocuments%2FWACV24%5FPeople%5FFlow&p=14)-
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
