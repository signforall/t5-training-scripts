import os
import torch
import numpy as np
import random
from transformers import TrainingArguments, Trainer, AutoTokenizer
from modules.dataset import VideoDataset
from modules.model import SNLTraslationModel
from modules.config_loader import load_config
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import evaluate

def load_trained_model(cfg):

    tokenizer = AutoTokenizer.from_pretrained(cfg.ModelArguments.t5_model_name)
    model = SNLTraslationModel(
        feature_dim=cfg.ModelArguments.feature_dim, 
        hidden_dim=cfg.ModelArguments.hidden_dim, 
        model_name=cfg.ModelArguments.base_model_name, 
        base_model_weight=cfg.ModelArguments.t5_model_name
    )

    model.load_state_dict(torch.load(f"{cfg.ModelArguments.checkpoint_path}/pytorch_model.bin", map_location="cpu"))

    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model, tokenizer

def collate_fn(batch):
    features = torch.stack([item["features"] for item in batch]).cuda()
    attention_mask = torch.stack([item["attention_mask"] for item in batch])[:, :, 0].cuda()
    return {"features": features, "attention_mask": attention_mask, "labels":[item["labels"] for item in batch]}

def main():
    # Load configuration
    cfg = load_config()

    # Load model and tokenizer
    model, tokenizer = load_trained_model(cfg)

    # Load metric
    sacrebleu = evaluate.load('sacrebleu')

    # Load datasets
    test_dataset = VideoDataset(h5_file_path=cfg.DatasetArguments.test_dataset_path, \
                                label_file_path=cfg.DatasetArguments.test_labels_dataset_path, \
                                tokenizer=tokenizer, \
                                test_set=True)

    # Create DataLoader for batching
    batch_size = 16  # Adjust batch size based on available GPU memory
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    predictions = []
    references = []
    
    os.makedirs(cfg.EvaluationArguments.results_save_path, exist_ok=True)
    prediction_log_file = f"{cfg.EvaluationArguments.results_save_path}/{cfg.ModelArguments.checkpoint_path.split(os.sep)[-1].split('.')[0]}_result.txt"

    with torch.no_grad():
        with open(prediction_log_file, "w") as f_pred:
            for batch in tqdm(test_loader, desc="Evaluating..."):
                
                # Pass through linear layers
                x = model.custom_linear(batch["features"])
                output = model.model.generate(inputs_embeds=x, attention_mask=batch["attention_mask"], do_sample=False)


                # Decode predictions
                batch_preds = [tokenizer.decode(o, skip_special_tokens=True) for o in output]
                
                # Store results
                predictions.extend(batch_preds)
                references.extend(batch["labels"])

                # Write batch predictions to file
                for ref, pred in zip(batch["labels"], batch_preds):
                    f_pred.write(f"reference_text: {ref}\ngenerated_text: {pred}\n{'-' * 10}\n")
    

            result = sacrebleu.compute(predictions=predictions, references=references)
            result = {
                    "bleu": result["score"], 
                    'bleu-1': result['precisions'][0],
                    'bleu-2': result['precisions'][1],
                    'bleu-3': result['precisions'][2],
                    'bleu-4': result['precisions'][3],
                }

            f_pred.write(f"sacrebleu\n")
            f_pred.write(f"BLEU score: {result['bleu']}"+"\n")

            f_pred.write(f"BLEU-1: {result['bleu-1']}"+"\n")
            f_pred.write(f"BLEU-2: {result['bleu-2']}"+"\n")
            f_pred.write(f"BLEU-3: {result['bleu-3']}"+"\n")
            f_pred.write(f"BLEU-4: {result['bleu-4']}"+"\n")


if __name__=='__main__':
    main()