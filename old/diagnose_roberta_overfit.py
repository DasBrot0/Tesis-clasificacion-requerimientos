"""Diagnostic only: can the RoBERTa-BNE mirror memorize 48 training examples?"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from security_multilabel_common import LABEL_COLS, SEED, TEXT_COL, seed_everything

MODEL_SOURCE = "Buzzeitor/roberta-base-bne"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(SEED)

data = pd.read_csv("dataset_seguridad_provisional.csv", encoding="utf-8-sig")
manifest = pd.read_csv("artifacts/security_split_manifest_seed42.csv", encoding="utf-8-sig")
train = data.merge(manifest[["ID", "split"]], on="ID").query("split == 'train'").copy()

rare = train[train[["Non_repudiation", "Accountability", "Resistance"]].any(axis=1)]
other_positive = train[train["Security"].eq(1) & ~train["ID"].isin(rare["ID"])].sample(
    n=min(24 - len(rare), len(train)), random_state=SEED
)
positive = pd.concat([rare, other_positive]).drop_duplicates("ID").head(24)
negative = train[train["Security"].eq(0)].sample(n=24, random_state=SEED)
subset = pd.concat([positive, negative]).sample(frac=1, random_state=SEED).reset_index(drop=True)
assert len(subset) == 48

tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE)

class TinyDataset(Dataset):
    def __len__(self): return len(subset)
    def __getitem__(self, index):
        row = subset.iloc[index]
        encoded = tokenizer(str(row[TEXT_COL]), truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(row[LABEL_COLS].to_numpy(dtype=np.float32))
        return item

config = AutoConfig.from_pretrained(MODEL_SOURCE, num_labels=len(LABEL_COLS), problem_type="multi_label_classification")
config.hidden_dropout_prob = 0.1
config.attention_probs_dropout_prob = 0.1
config.classifier_dropout = 0.1
model = AutoModelForSequenceClassification.from_pretrained(MODEL_SOURCE, config=config).to(DEVICE)
loader = DataLoader(TinyDataset(), batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(SEED))
positive_counts = subset[LABEL_COLS].sum().to_numpy(dtype=float)
pos_weight = torch.tensor((len(subset) - positive_counts) / np.maximum(positive_counts, 1), dtype=torch.float32, device=DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.0)
history = []

for epoch in range(1, 51):
    model.train(); total = 0.0
    for batch in loader:
        labels = batch.pop("labels").to(DEVICE)
        inputs = {key: value.to(DEVICE) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        logits = model(**inputs).logits
        loss = criterion(logits, labels)
        loss.backward(); optimizer.step()
        total += loss.item() * len(labels)
    model.eval(); truths=[]; probs=[]
    with torch.no_grad():
        for batch in DataLoader(TinyDataset(), batch_size=16, shuffle=False):
            labels = batch.pop("labels").to(DEVICE)
            inputs = {key: value.to(DEVICE) for key, value in batch.items()}
            truths.append(labels.cpu().numpy())
            probs.append(torch.sigmoid(model(**inputs).logits).cpu().numpy())
    y_true=np.vstack(truths); y_pred=(np.vstack(probs)>=0.5).astype(int)
    per_label_f1=[]
    from sklearn.metrics import f1_score
    macro=float(f1_score(y_true,y_pred,average="macro",zero_division=0))
    exact=float((y_true==y_pred).all(axis=1).mean())
    row={"epoch":epoch,"train_loss":total/len(subset),"train_macro_f1":macro,"train_subset_accuracy":exact}
    history.append(row)
    if epoch == 1 or epoch % 5 == 0:
        print(row, flush=True)

out = Path("artifacts/roberta_bne_overfit_diagnostic.csv")
pd.DataFrame(history).to_csv(out, index=False, encoding="utf-8-sig")
summary = {
    "model_source": MODEL_SOURCE,
    "subset_size": len(subset),
    "epochs": 50,
    "learning_rate": 5e-5,
    "max_length": 64,
    "used_weighted_bce": True,
    "final": history[-1],
    "minimum_train_loss": min(row["train_loss"] for row in history),
    "maximum_train_macro_f1": max(row["train_macro_f1"] for row in history),
    "maximum_train_subset_accuracy": max(row["train_subset_accuracy"] for row in history),
}
Path("artifacts/roberta_bne_overfit_diagnostic.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(summary)
