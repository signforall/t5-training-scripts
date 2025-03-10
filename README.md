# Sign Language Translation Using T5
This repository contains the training and evaluation scripts for training Saudi Sign Language using T5 variations.

## Pre-requisites
The training and inference were conducted using the Azure image: `acpt-pytorch-2.2-cuda12.1` (version 26), then packages in `requirements.txt` were installed.

- Python version: 3.10  
- PyTorch version: 2.2  

Rather than using the same image, you can manually install Python and PyTorch, then install the required dependencies:
```bash
pip install -r requirements.txt
```

## Training
1. Locate the `config_train.yaml` file in the `configs` folder.
2. Edit the file as needed, specifying the locations of the training and validation datasets.
3. Run the training using the following command:
   ```bash
   python train.py --config_path configs/config_train.yaml
   ```

### Fine-tuning with YouTubeASL Checkpoint
If you want to fine-tune using the YouTubeASL checkpoint:
1. Download the checkpoint from [here](https://drive.google.com/drive/folders/1TM1BrA6v4bJTd0rzSHFUp0yH-FmXO9nK?usp=drive_link).
2. Set `resume` to `true` in `config_train.yaml`.
3. Specify the checkpoint path in `resume_checkpoint`.

## Evaluation
1. Specify the location of the checkpoint in the `config_eval.yaml` file.
2. Edit the file as needed, and specify the path of the generated results.
3. Run the evaluation using the following command:
   ```bash
   python eval.py --config_path configs/config_eval.yaml
   ```
### Inference Without Labels
To generate results for a test set without labels, simply leave `test_labels_dataset_path` empty in the evaluation config file.
