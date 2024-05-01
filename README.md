[![Python 3.8.10](https://img.shields.io/badge/python-3.8.10+-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3810/)
[![NumPy](https://img.shields.io/badge/numpy-1.24.3+-green?logo=numpy&logoColor=white)](https://pypi.org/project/numpy/1.24.3/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-3.7.1+-green?logo=plotly&logoColor=white)](https://pypi.org/project/matplotlib/3.7.1)
[![Notebook](https://img.shields.io/badge/notebook-6.5.4+-green?logo=jupyter&logoColor=white)](https://pypi.org/project/notebook/6.5.4)
[![torch](https://img.shields.io/badge/torch-2.0.0+-green?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![torchvision](https://img.shields.io/badge/torchvision-0.15.1+-green?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![pandas](https://img.shields.io/badge/pandas-2.0.1+-green?logo=pandas&logoColor=white)](https://pypi.org/project/pandas/2.0.1/)

# From Posterior Sampling to Meaningful Diversity in Image Restoration [ICLR 2024]

| [Project page](https://noa-cohen.github.io/MeaningfulDiversityInIR/) | [arXiv](https://arxiv.org/abs/2310.16047) | 

![Teaser](https://github.com/noac-github/MeaningfulDiversity/blob/main/Teaser.jpg?raw=true)


## Table of Contents

- [Requirements](#requirements)
- [Usage Example - Guided Meaningful Diversity](#usage-example---guided-meaningful-diversity)
    - [RePaint](#repaint)
    - [DDRM](#ddrm)
    - [DDNM](#ddnm)
    - [DPS](#dps)
- [Usage Example - Baseline Methods](#usage-example---baseline-methods)
  - [1. Generate images](#1-generate-images)
  - [2. Extract features](#2-extract-features)
  - [3. Sub-sample using a baseline approach](#3-sub-sample-using-a-baseline-approach)
- [Citation](#citation)

## Requirements

```bash
python -m pip install -r requirements.txt
```

For running the baseline methods you should also install [faiss](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md).


## Usage Example - Guided Meaningful Diversity

We apply guidance on RePaint, DDRM, DDNM and DPS. We therefore add their code with our guidance implemented in it.  
To run using this code, you need to supply the models the relevant checkpoint. For fair comparison we use the same checkpoints for all models where applicable, that can be downloaded here:

- [256x256 unconditional imagenet checkpoint](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt) from [Guided Diffusion](https://github.com/openai/guided-diffusion)
- [256x256 CelebAMask-HQ](https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt) from [SDEdit](https://github.com/ermongroup/SDEdit) for DDRM, DDNM, DPS
- [256x256 CelebA-HQ](https://drive.google.com/uc?id=1norNWWGYP3EZ_o05DmoW1ryKuKMmhlCX) from [RePaint](https://github.com/andreas128/RePaint) for RePaint

Place the downloaded checkpoint in the `checkpoints` folder, under the names `256x256_diffusion_uncond.pt`, `celeba_hq.ckpt` and `celeba256_250000.pt`, respectively.
The folder path can be changed either by updating the config yaml (RePaint, DPS) or by setting a different `--checkpoints_folder` (DDNM, DDRM).
Then, each method uses it's own running configurations.  
In all methods, for the guidance add:

```bash
--guidance_dist <guidance_D_val> --guidance_eta <guidance_eta_val>
```

to set the guidance parameters, $\eta$ and $D$. Running without `--guidance_dist`, `--guidance_eta` will lead to vanilla posterior sampling.  
The size of the sampled set, $N$, is determined using `--parallel_batch` in all methods, and is set to `5` as the default.  

### RePaint
Original repo: [https://github.com/andreas128/RePaint](https://github.com/andreas128/RePaint)
by Huawei Technologies Co., Ltd. is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

In `diverse_models/RePaint-guided/configs` we include two possible yaml files, for CelebA and ImageNet.  

```bash
cd diverse_models/RePaint-guided
python test.py --conf_path <config_yaml>
```

- The noise amount is deterimned in `noise.sigma` in the yaml file.
- The ground truth images path is set in `data.eval.inpainting_config.gt_path` in the yaml file.
- The masks images path is set in `data.eval.inpainting_config.mask_path` in the yaml file.
- The output directory paths are set under `data.eval.inpainting_config.paths` in the yaml file.

### DDRM
Original repo: [https://github.com/bahjat-kawar/ddrm](https://github.com/bahjat-kawar/ddrm)

In `diverse_models/DDRM-guided/configs` we include two possible yaml files, for CelebA and ImageNet.  

```bash
cd diverse_models/DDRM-guided
python main.py --config <domain_config_yaml> --timesteps 20 --deg sr_bicubic16 -i <out_dir_path>  --sigma_0 <noise_level> --path_y <path_to_gt_images>
```

- Set `--sigma_0` to control noisy vs. noiseless super-resolution.
- The number in `--deg` controls the degradation factor

### DDNM
Original repo: [https://github.com/wyhuai/DDNM](https://github.com/wyhuai/DDNM)

In `diverse_models/DDNM-guided/configs` we include two possible yaml files, for CelebA and ImageNet.  

#### Super resolution

```bash
cd diverse_models/DDNM-guided
python main.py --config <diffusion_config_yaml> --deg sr_bicubic --deg_scale <deg_scale> --path_y <path_to_gt_images> -i <out_dir_path> --parallel_batch <N>
```

- Add `--sigma_y <noise_level> --add_noise` for noisy super-resolution.
- `--deg_scale` controls the degradation factor

#### Image Inpainting

```bash
cd diverse_models/DDNM-guided
python main.py --ni --config <diffusion_config_yaml> --deg inpainting --path_y <path_to_gt_images> --path_masks <path_to_mask_images> -i <out_dir_path> --parallel_batch <N>
```

### DPS
Original repo: [https://github.com/DPS2022/diffusion-posterior-sampling](https://github.com/DPS2022/diffusion-posterior-sampling)

In `diverse_models/DPS-guided/configs` we include six possible yaml configurations, for CelebA and ImageNet, on inpainting and super-resolution (both noisy and noiseless).  

```bash
cd diverse_models/DPS-guided
python sample_condition.py --model_config <model_config_yaml> --diffusion_config <diffusion_config_yaml> --task_config <task_config_yaml> --save_dir <out_dir_path>
```

- The noise amount is deterimned under `noise.sigma` in the yaml file.
- The ground truth images path is set in `data.root` in the yaml file.
- The masks images path is set in `measurement.mask_opt.mask_path` in the yaml file.

## Usage Example - Baseline Methods

### 1. Generate Images

First generate $\tilde{N}$ samples from your favorite vanilla posterior sampler. Place them in the following structure (this is the convention of all posterior samplers here in the repo):

```bash

├── ImageName
|   └── images
|   |	├── 0_ImageName.jpg
|   |	├── 1_ImageName.jpg
|   |   |   ...
|   |	└── N-1_ImageName.jpg
└── Other data files
```

### 2. Extract features

For the faces domain, or for running K-means on ImageNet, extract features using `feature_extraction.py`

```bash
cd BaselineApproaches
python feature_extraction.py --im_dir <path to ImageName folder> --domain <faces|inet> 
```

- To use on image inpainting, add `--inpainting` and provide `--mask_path`.

For running ImageNet with FPS or Uniformizaition, extract features using `extract_distances_imagenet.py`

```bash
cd BaselineApproaches
python extract_distances_imagenet.py --im_dir <path to ImageName folder> --domain <faces|inet> 
```

### 3. Sub-sample using a baseline approach

After features were extracted, run `sampling.py` to sub-sample.

```bash
cd BaselineApproaches
python sampling.py --approach <approach_name> --im_dir <ImageName> --domain <faces|inet> --feature_type <patch|deep_features> --
```

- `--approach` can take one of the three explored baselines, `kmeans` for K-Means, `unif` for Uniformization, or `fps` for FPS.
- `--feature_type` can be either `patch` for using the restored pixels in the mask region of the image or `deep_features` for using the features extracted in step B.
- To use on image inpainting, add `--inpainting`.

## Citation

If you use this code for your research, please cite our paper:

```bash
@inproceedings{
  cohen2024from,
  title={From Posterior Sampling to Meaningful Diversity in Image Restoration},
  author={Noa Cohen and Hila Manor and Yuval Bahat and Tomer Michaeli},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=ff2g30cZxj}
}
```
