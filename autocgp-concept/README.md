## Automatic Concept Discovery

#### 🛠️ Environment Installation

This codebase has been tested on the following configuration:
*   **GPU:** NVIDIA GeForce RTX 3090
*   **Driver:** Version 535.247.01
*   **CUDA:** Version 12.2

```
# Create the environment from the provided YAML file
conda env create -f environment.yml

# Activate the environment
conda activate autocgp-concept
```

#### 📂 Dataset

**[TODO: Add Dataset Instructions]**

#### 🚀 Training

Before running the training scripts, you must configure the dataset path in `AutoCGP/autocgp-concept/src/path.py`. Set the `DATA_PATH` variable to the path of your dataset directory.

To train the model, use the provided shell script.

**Note:** You can modify hyperparameters by editing `train.sh` or by passing arguments directly. Key arguments are defined in `autocgp-concept/src/train_multi.py`.

```
bash train.sh
```

#### 🏷️ Labeling

After training, you can generate concept labels.

**Note:** Argument definitions for labeling can be found in `autocgp-concept/src/label_all_multi.py`.

```
bash label.sh
```

The concept labels, visualizations, and embeddings will be stored in the `model_checkpoints` directory.

```
autocgp-concept/src/model_checkpoints/[log_directory]/
├── [task_name]/
│   ├── key_[checkpoint_epoch_index]/
│   │   ├── eval/                   # Evaluation data split results
│   │   ├── eval.txt                # Evaluation data split summary
│   │   ├── train.txt               # Training data split results
│   │   └── train/
│   │       ├── [concept_index_1]/  # Folder for a specific discovered concept
│   │       │   ├── [video_id].mp4  # Video clip demonstrating the concept
│   │       │   └── ...
│   │       ├── [concept_index_2]/
│   │       ├── ...
│   │       ├── [demo_index]_emb.npy    # Embeddings (Numpy)
│   │       ├── [demo_index]_label.npy  # Concept labels (Numpy)
│   │       └── ...
└── ...
```
