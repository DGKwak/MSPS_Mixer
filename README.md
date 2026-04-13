# MSPS Mixer

MSPS Mixer is a multiscale patch shift mixer model for micro-Doppler based 6-class activity recognition. It supports both standard training and knowledge distillation (KD).

## Key Features

- Multiscale patch embedding based classification
- Standard training pipeline: Main.py
- Knowledge distillation pipeline: Main_KD.py
- Hydra-based configuration management
- Automatic saving of training logs, checkpoints, confusion matrices, and CSV results

## Requirements

- Python 3.12 or later
- A CUDA-capable GPU is recommended
- Package manager: uv

## Installation

Run the following command from the project root:

```bash
uv sync
```

If needed, activate the virtual environment first:

```bash
source .venv/bin/activate
```

## Dataset Structure

The code uses the ImageFolder format. The folder structure should look like this:

```text
data/STFT_PSplit/
  train/
    Drinking/
    Falling/
    Picking/
    Sitting/
    Standing/
    Walking/
  val/
    Drinking/
    Falling/
    Picking/
    Sitting/
    Standing/
    Walking/
  test/
    Drinking/
    Falling/
    Picking/
    Sitting/
    Standing/
    Walking/
```

For KD training, the student and teacher use different input sizes:

- Student input size: 96
- Teacher input size: 224

## Dataset

This project uses the University of Glasgow research dataset titled Radar signatures of human activities.

- Official dataset page: https://researchdata.gla.ac.uk/848/
- DOI: 10.5525/gla.researchdata.848
- License: CC BY 4.0

According to the dataset description, it contains radar signatures of different indoor human activities performed by different people in different locations. It is intended for developing and benchmarking feature extraction and classification methods in assisted living scenarios, such as fall detection and activity anomaly detection.

The dataset used in this project is a preprocessed STFT-based split of that radar activity data, organized into the following six classes:

- Drinking
- Falling
- Picking
- Sitting
- Standing
- Walking

If you use the dataset in your work, please cite the original University of Glasgow research data record and its related publication.

## How to Run

### Standard Training

```bash
uv run Main.py
```

### Knowledge Distillation Training

```bash
uv run Main_KD.py
```

You can override Hydra settings at runtime if needed:

```bash
uv run Main.py epochs=300 batch_size=16 learning_rate=1e-4
uv run Main_KD.py epochs=300 lambda_kl=0.5 temperature=2
```

## Configuration Files

### Standard Training

- Main config: config/main.yaml
- Model config: config/model/MSPS_Mixer.yaml
- Data config: config/data/STFT_PSplit.yaml

### KD Training

- Main config: config/main_KD.yaml
- Model config: config/model/MSPS_Mixer_KD.yaml
- Data config: config/data/STFT_KD.yaml

Main hyperparameters:

- Epochs: 200
- Batch size: 32
- Learning rate: 1e-3
- Weight decay: 1e-4
- Random seed: 2024

Additional KD hyperparameters:

- lambda_kl: 0.75
- temperature: 1

## Comparison with Existing Methods

The table below compares this work with representative prior models using two metrics: Test Accuracy and Number of Parameters.

| Model            | Accuracy (%) | Number of Parameters (M) |
| ---------------- | -----------: | -----------------------: |
| EfficientNet     |        90.96 |                      4.0 |
| MobileNetv2      |        92.09 |                      2.2 |
| DeiT             |        89.26 |                      5.5 |
| MobileViT        |        88.13 |                      1.0 |
| LH-ViT           |        92.10 |                      0.7 |
| MSPS-Mixer       |        94.91 |                      1.6 |
| MSPS-Mixer w/ KD |        93.22 |                      0.3 |

Summary:

- MSPS-Mixer achieves the best Test Accuracy among the compared models.
- MSPS-Mixer w/ KD significantly reduces the parameter count while preserving competitive accuracy.

## Output Locations

After training, the following outputs are saved:

- CSV results: results/
- Confusion matrices: results/confusion/
- Best model checkpoints: checkpoints/

File names include the execution timestamp.

## Project Structure

- Main.py: Standard training entry point
- Main_KD.py: Knowledge distillation training entry point
- trainer.py: Training and evaluation loops
- model/: Model definitions
- loss/: Loss functions
- utils/: Data loading, early stopping, and helper utilities

## Notes

- Input images are automatically resized according to the configured transforms.Resize size.
- CUDA is used automatically if a GPU is available.
- Testing is performed using the saved best checkpoint.

## Common Commands

```bash
uv run Main.py
uv run Main_KD.py
```

## License

If no separate license file is provided, follow the project owner's usage guidance.
