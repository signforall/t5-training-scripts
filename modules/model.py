import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, MT5ForConditionalGeneration

class SNLTraslationModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim, model_name, base_model_weight):
        super(SNLTraslationModel, self).__init__()

        if model_name == 'mT5':
            self.model = MT5ForConditionalGeneration.from_pretrained(base_model_weight)
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(base_model_weight)

        self.custom_linear = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Dropout(0),
            nn.GELU(),
        )

    def forward(self, features, attention_mask, labels):
        x = self.custom_linear(features)
        attention_mask = attention_mask[:, :, 0]
        outputs = self.model(inputs_embeds=x, attention_mask=attention_mask, labels=labels, return_dict=True)
        return outputs.loss, outputs.logits
