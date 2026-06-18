# Trajectory Prompt Learning for Object Motion Direction Reasoning in Videos

This repository is a modified implementation of **STOP: Integrated Spatial-Temporal Dynamic Prompting for Video Understanding**, adapted for **object motion direction reasoning** in videos.

The original STOP framework focuses on spatial-temporal prompt learning for video understanding. In this project, we extend the STOP-based prompt learning structure to a **direction classification task**, where the model predicts the movement direction of an object from a short video.

## Overview

Recent Vision-Language Models show strong performance on image understanding tasks, but they still struggle with video reasoning tasks that require temporal and spatial changes to be understood together. Object motion direction reasoning is one such task: the model must compare multiple frames and infer how the object position changes over time.

To address this, this project introduces a **Trajectory Prompt** module that explicitly captures frame-to-frame feature changes and injects trajectory-level information into the video representation.

## Key Idea

The main idea is simple:

> Instead of relying only on static frame features, we add a trajectory prompt that summarizes how visual features change across frames.

For each video, we extract frame-level visual features from the CLIP-based video encoder. Then we compute the feature differences between adjacent frames and accumulate them to represent the overall movement trend.

The trajectory feature is summarized using:

* Initial movement information
* Global cumulative trajectory trend
* Final cumulative movement information

These features are passed through an MLP to generate a trajectory prompt.

## Method

### 1. Direction Classification

The original STOP structure is mainly designed for video-text matching and retrieval-style tasks. In this project, we modify the training objective for **9-way direction classification**.

The model predicts one of the following classes:

* Upper Right
* Up
* Upper Left
* Left
* Stationary
* Right
* Lower Left
* Down
* Lower Right

Instead of using a retrieval similarity objective, the model is trained with a classification loss based on the ground-truth direction label.

### 2. Trajectory Prompt

Given frame-level features, we compute the difference between adjacent frames:

```text
Δh_t = h_(t+1) - h_t
```

Then cumulative trajectory features are computed as:

```text
C_t = Σ Δh_k,  k = 1 ... t
```

The final trajectory representation is constructed by concatenating:

```text
z_traj = concat(C_first, C_global, C_last)
```

where:

* `C_first` represents the initial movement
* `C_global` represents the average cumulative trajectory
* `C_last` represents the final accumulated movement

The trajectory feature is then passed through an MLP:

```text
p_traj = MLP(z_traj)
```

Finally, a prompt scale is applied:

```text
p_traj = α · p_traj
```

In our final setting, we use:

```text
α = 0.005
```

## Dataset

We construct a synthetic direction dataset for object motion direction reasoning.

Each sample contains:

* A short video
* A direction label
* A 9-way direction class annotation

The dataset is designed so that the correct label depends directly on object position changes across frames. Therefore, the model must use temporal information rather than relying only on static appearance.

Dataset statistics:

| Split | Number of samples |
| ----- | ----------------: |
| Train |            28,701 |
| Test  |             7,175 |

Each video is sampled into 12 frames.

## Results

### Direction Classification

| Model setting                                    | Accuracy (%) |
| ------------------------------------------------ | -----------: |
| STOP baseline                                    |        37.30 |
| Trajectory Prompt with CLS feature               |        36.11 |
| Trajectory Prompt with token mean, scale = 0.05  |        42.27 |
| Trajectory Prompt with token mean, scale = 0.005 |    **75.48** |

The final model using token-level trajectory features and a prompt scale of `0.005` achieves **75.48% accuracy**, which is more than twice the baseline performance.

### Analysis

The baseline model shows a strong bias toward the `Stationary` class. It often predicts moving objects as stationary, indicating that it does not reliably capture object movement.

The CLS-based trajectory prompt reduces false stationary predictions, but it does not improve fine-grained direction classification. This suggests that CLS features can help detect whether movement exists, but they are not sufficient for distinguishing detailed motion directions.

The token-mean trajectory prompt performs much better because it preserves more spatial information from frame-level patch tokens. With an appropriate prompt scale, it improves both movement detection and direction classification.

### Additional MSR-VTT Experiment

We also evaluate the proposed structure on MSR-VTT to check whether adding the trajectory prompt harms general video-text retrieval performance.

| Model                    |  R@1 |
| ------------------------ | ---: |
| CLIP4Clip                | 43.1 |
| STOP baseline            | 46.9 |
| STOP + Trajectory Prompt | 45.7 |

The trajectory prompt slightly decreases R@1 compared to the directly reproduced STOP baseline, but it still maintains a similar level of retrieval performance. This suggests that the proposed module improves direction reasoning while not severely damaging general video-text representation.

## Environment

This code is based on:

* PyTorch 2.4.0
* CUDA 11.8
* torchvision 0.19.0

To create the environment:

```bash
conda env create -f environment.yaml
conda activate STOP
```

## Download CLIP Model

Download the CLIP ViT-B/32 pretrained weights and place them in your pretrained model directory.

```bash
wget https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
```

Then update the pretrained model path in the corresponding script or config file.

## Data Preparation

For video preprocessing, use:

```bash
python preprocess/compress_video.py \
  --input_root [raw_video_path] \
  --output_root [compressed_video_path]
```

The preprocessing script compresses videos to 3 FPS and resizes them to width or height 224.

## Training

### Direction Classification

Example command:

```bash
bash scripts/direction_mcq.sh
```

or run `main.py` directly with the required arguments:

```bash
python main.py \
  --dataset direction_mcq \
  --use_traj_prompt 1 \
  --traj_prompt_scale 0.005 \
  --traj_feat_type token_mean \
  --freeze_clip 1
```

Please update dataset paths and training arguments according to your local environment.

### MSR-VTT

To train or evaluate on MSR-VTT:

```bash
bash scripts/msrvtt.sh
```

## Repository Structure

```text
STOP/
├── dataloaders/        # Dataset loaders
├── dataset/            # Dataset-related files
├── figs/               # Figures
├── modules/            # Model and prompt modules
├── preprocess/         # Video preprocessing scripts
├── scripts/            # Training and evaluation scripts
├── utils/              # Utility functions
├── main.py             # Main training/evaluation entry
├── params.py           # Argument parser
├── environment.yaml    # Conda environment
└── README.md
```

## Main Modifications

Compared to the original STOP implementation, this project includes:

* Direction classification formulation
* Synthetic direction dataset support
* Trajectory prompt module
* CLS-based and token-based trajectory feature comparison
* Prompt scale analysis
* Additional MSR-VTT verification

## Acknowledgement

This repository is based on the official implementation of:

**STOP: Integrated Spatial-Temporal Dynamic Prompting for Video Understanding**

We also acknowledge the original STOP, CLIP4Clip, and related video-language understanding works.

## Citation

If you use the original STOP code, please cite:

```bibtex
@article{liu2025stop,
  title={STOP: Integrated Spatial-Temporal Dynamic Prompting for Video Understanding},
  author={Liu, Zichen and Xu, Kunlun and Su, Bing and Zou, Xu and Peng, Yuxin and Zhou, Jiahuan},
  journal={arXiv preprint arXiv:2503.15973},
  year={2025}
}
```

If you use this modified version, please also mention that it adapts STOP for object motion direction reasoning with trajectory prompt learning.
