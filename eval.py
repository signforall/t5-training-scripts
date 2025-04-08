import os
import torch
import pandas as pd
from transformers import AutoTokenizer
from modules.dataset import VideoDataset
from modules.model import SNLTraslationModel
from modules.config_loader import load_config
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sacrebleu.metrics import BLEU

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
    features = torch.stack([item["features"] for item in batch]).to("cuda" if torch.cuda.is_available() else "cpu")
    attention_mask = torch.stack([item["attention_mask"] for item in batch])[:, :, 0].to("cuda" if torch.cuda.is_available() else "cpu")
    return {"features": features, "attention_mask": attention_mask,\
             "labels": [item["labels"] for item in batch], "keys": [item["key"] for item in batch]}

def main():
    # Load configuration
    cfg = load_config()

    # Load model and tokenizer
    model, tokenizer = load_trained_model(cfg)

    # # Load metric
    # sacrebleu = evaluate.load('sacrebleu')

    # Load datasets
    if not os.path.exists(cfg.DatasetArguments.test_labels_dataset_path):
        with_labels = False
    else:
        with_labels = True

    test_dataset = VideoDataset(h5_file_path=cfg.DatasetArguments.test_dataset_path, \
                                label_file_path=cfg.DatasetArguments.test_labels_dataset_path, \
                                tokenizer=tokenizer, \
                                test_set=True, \
                                with_labels=with_labels)

    # Create DataLoader for batching
    batch_size = cfg.EvaluationArguments.batch_size  # Adjust batch size based on available GPU memory
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    keys = []
    predictions = []
    if with_labels:
        references = []
    
    os.makedirs(cfg.EvaluationArguments.results_save_path, exist_ok=True)
    prediction_save_file = f"{cfg.EvaluationArguments.results_save_path}/{cfg.ModelArguments.checkpoint_path.split(os.sep)[-1].split('.')[0]}_result.csv"

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating..."):
            
            # Pass through linear layers
            x = model.custom_linear(batch["features"])
            output = model.model.generate(inputs_embeds=x, attention_mask=batch["attention_mask"], do_sample=False)

            # Decode predictions
            batch_preds = [tokenizer.decode(o, skip_special_tokens=True) for o in output]
            
            # Store results
            keys.extend(batch["keys"])
            predictions.extend(batch_preds)
            if with_labels:
                references.extend(batch["labels"])  

    if with_labels:
        df_results = pd.DataFrame({'ID': keys, \
                                'Translation': predictions, 'Reference': references})
    else:
        df_results = pd.DataFrame({'ID': keys, \
                                'Translation': predictions})
    
    df_results.to_csv(prediction_save_file, encoding='utf-8', index=False)

    if not with_labels:
        exit(1)


    # result = sacrebleu.compute(predictions=predictions, references=references)
    
    bleu1 = BLEU(max_ngram_order=1).corpus_score(predictions,  references)
    bleu2 = BLEU(max_ngram_order=2).corpus_score(predictions,  references)
    bleu3 = BLEU(max_ngram_order=3).corpus_score(predictions,  references)
    bleu4 = BLEU(max_ngram_order=4).corpus_score(predictions,  references)
    
    # result = {
    #         'bleu-1': result['precisions'][0],
    #         'bleu-2': result['precisions'][1],
    #         'bleu-3': result['precisions'][2],
    #         'bleu-4': result['precisions'][3],
    #     }

    result = {
        'bleu-1': bleu1,
        'bleu-2': bleu2,
        'bleu-3': bleu3,
        'bleu-4': bleu4,
    }
            
    print('='*30)
    print("sacrebleu")
    print('-'*10)
    print(f"BLEU-1: {result['bleu-1']}")
    print(f"BLEU-2: {result['bleu-2']}")
    print(f"BLEU-3: {result['bleu-3']}")
    print(f"BLEU-4: {result['bleu-4']}")
    print('='*30)


if __name__=='__main__':
    main()
