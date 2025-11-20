# Concept-Guided Policy Learning for Robotic Manipulation (ICLR 2025)

This repository presents the code implementation for the ICLR 2025 research project "Concept-Guided Policy Learning for Robotic Manipulation." Our work focuses on leveraging concept guidance for policy learning through a two-stage approach: first, concept discovery to obtain concept files and generated masks, and then utilizing this information to train and evaluate robotic manipulation policies.

## Introduction

Effectively learning and generalizing policies in robotic manipulation tasks is a central challenge. This project introduces a **Concept-Guided** policy learning framework that aims to extract high-level semantic concepts during a **Concept Discovery** phase and utilize these concepts to guide the **Policy Learning** process. This approach enhances the interpretability, learning efficiency, and generalization capabilities of the learned policies.

Our methodology primarily involves the following two stages:

1.  **Concept Discovery**: In this stage, we extract high-level semantic concepts from raw data and generate corresponding `concept files` and `generated masks`. 
2.  **Policy Learning & Evaluation**: Using the concept files and masks generated in the first stage, we train a robotic manipulation policy. This repository contains the code for policy training and subsequent evaluation in the MimicGen simulation environment.

## Project Structure (Current Repository)

This repository primarily focuses on code related to policy learning and evaluation.

*   `scripts/train_diffusion_concept.py`: The main script for policy training.
*   `scripts/eval.py`: The main script for policy evaluation.
*   `config/`: Contains YAML configuration files for training and evaluation.
*   `utilsxs/`: Includes modules for data loading, model definition, and utility functions.

## Installation

Follow these steps to install the project:

```bash
# Assuming you are in the project root directory ./autocgp
# Create and activate the virtual environment
conda env create -f environment.yml
conda activate autocgp
pip install -e . 
```

## Usage

### 1. Concept Discovery

Check directory `autocgp-concept`

### 2. Policy Training

After generating the concept files and masks, you can use the `scripts/train.sh` script to train the policy.

```bash
python train_diffusion_concept.py \
    save_dir="your_save_dir" \
    dataset.data_dirs="['your_dataset_path']" \
    dataset.proto_dirs="your_concept_path" \
    dataset.mask="your_generated_mask_path"
```

*   `save_dir`: Specifies the directory to save model checkpoints, and relavant information.
*   `dataset.data_dirs`: Specifies the path to your training dataset, e.g., `['/path/to/your/dataset']`.
*   `dataset.proto_dirs`: Specifies the path to the concept files, which are outputs from the concept discovery stage.
*   `dataset.mask`: Specifies the path to the generated mask files, also outputs from the concept discovery stage.

### 3. Policy Evaluation

Once the policy is trained, you can use the `scripts/eval.sh` script to evaluate its performance in the MimicGen simulation environment.

```bash
python eval.py \
    --pretrain_path="your_ckpt_path" \
    --ckpt_number="your_ckpt_number"
```

*   `--pretrain_path`: Specifies the directory where your pretrained model checkpoints are located, e.g., `your_save_dir/run_name`.
*   `--ckpt_number`: Specifies the checkpoint number to load (e.g., if the file is `ckpt_99.pt`, use `ckpt_number=99`).

Evaluation results (e.g., GIF videos and success rates) will be saved in the directory specified by `--pretrain_path`.

## Experimental Environment

Our experiments are conducted in the **MimicGen** simulation environment for robotic manipulation tasks.

## Acknowledgments

We would like to acknowledge the following projects which have significantly influenced this work:

*   **Diffusion Policy**: [https://github.com/real-stanford/diffusion_policy]
*   **XSkill**: [https://github.com/real-stanford/xskill]

## License

This project is licensed under the MIT License.
