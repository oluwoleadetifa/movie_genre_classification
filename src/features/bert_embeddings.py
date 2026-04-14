# src/features/bert_embeddings.py

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


class BertEmbedder:
    def __init__(self, model_name="distilbert-base-uncased", device=None, max_length=256):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_batch(self, texts):
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        outputs = self.model(**encoded)

        # Mean pooling
        last_hidden = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1)

        masked_hidden = last_hidden * attention_mask
        summed = masked_hidden.sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1)

        embeddings = summed / counts
        return embeddings.cpu().numpy()

    def encode_texts(self, texts, batch_size=16):
        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding text"):
            batch = list(texts[i:i + batch_size])
            emb = self.encode_batch(batch)
            all_embeddings.append(emb)

        return np.vstack(all_embeddings)
