import os
import torch
import numpy as np
import random
import wandb
from transformers import TrainingArguments, Trainer, AutoTokenizer
from modules.dataset import VideoDataset
from modules.model import SNLTraslationModel
from modules.config_loader import load_config

def set_seed(seed_value):
    random.seed(seed_value)  # Python random
    np.random.seed(seed_value)  # NumPy random
    torch.manual_seed(seed_value)  # PyTorch (CPU & CUDA)
    torch.cuda.manual_seed(seed_value)  # GPU-specific seed
    torch.cuda.manual_seed_all(seed_value)  # Multi-GPU safe

def main():
    # Load configuration
    cfg = load_config()

    # Set fixed seed for reproducibility
    set_seed(cfg.TrainigArguments.seed)

    # Set up experiment name
    EXP_NAME = f"SNL_Model-name_{cfg.ModelArguments.base_model_name}-Weights_{"BASE" if not cfg.ModelArguments.resume else "SNL_WEIGHTS"}-Language_{cfg.DatasetArguments.language}-seed_{cfg.TrainigArguments.seed}"

    # Set environment variables and Wandb
    os.environ["WANDB_API_KEY"] = cfg.WandbArguments.wandb_api_key
    wandb.init(
        project=cfg.WandbArguments.wandb_project,
        name=EXP_NAME
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.ModelArguments.base_model_hf)

    # Load datasets
    train_dataset = VideoDataset(cfg.DatasetArguments.train_dataset_path, cfg.DatasetArguments.train_labels_dataset_path, tokenizer, cfg.DatasetArguments.max_sequence_length)
    valid_dataset = VideoDataset(cfg.DatasetArguments.valid_dataset_path, cfg.DatasetArguments.valid_labels_dataset_path, tokenizer, cfg.DatasetArguments.max_sequence_length)

    # Load model
    model = SNLTraslationModel(
        feature_dim=cfg.ModelArguments.feature_dim, 
        hidden_dim=cfg.ModelArguments.hidden_dim, 
        model_name=cfg.ModelArguments.base_model_name, 
        base_model_weight=cfg.ModelArguments.base_model_hf
    )

    if cfg.ModelArguments.resume == True:
        print(f"------------ Loading {cfg.ModelArguments.resume_checkpoint}")
        try:
            state_dict = torch.load(f"{cfg.ModelArguments.resume_checkpoint}/pytorch_model.bin", map_location="cpu")
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error: {e}. Trying strict=False mode.")
            model.load_state_dict(state_dict, strict=False)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(cfg.TrainigArguments.output_folder, EXP_NAME),
        evaluation_strategy="epoch",
        learning_rate=cfg.TrainigArguments.learning_rate,
        per_device_train_batch_size=cfg.TrainigArguments.batch_size,
        gradient_accumulation_steps=cfg.TrainigArguments.gradient_accumulation_steps,
        num_train_epochs=cfg.TrainigArguments.epochs,
        report_to="wandb",
        weight_decay=cfg.TrainigArguments.weight_decay,
        save_strategy="epoch",
        per_device_eval_batch_size=cfg.TrainigArguments.per_device_eval_batch_size,
        eval_accumulation_steps=cfg.TrainigArguments.eval_accumulation_steps,
        fp16=cfg.TrainigArguments.fp16,
        seed=cfg.TrainigArguments.seed
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=lambda batch: {
            "features": torch.stack([sample['features'] for sample in batch]),
            "attention_mask": torch.stack([sample['attention_mask'] for sample in batch]),
            "labels": torch.stack([sample['labels'] for sample in batch])
        }
    )

    # Start training
    trainer.train()

if __name__=='__main__':
    main()