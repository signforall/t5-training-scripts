# README

## Pre-requisites
The training and inference were conducted using the Azure image: `acpt-pytorch-2.2-cuda12.1` (version 26).

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
1. Download the checkpoint from [link_here].
2. Set `resume` to `true` in `config_train.yaml`.
3. Specify the checkpoint path in `resume_checkpoint`.

## Evaluation
1. Specify the location of the checkpoint in the `config_eval.yaml` file.
2. Run the evaluation using the following command:
   ```bash
   python eval.py --config_path configs/config_eval.yaml
   ```