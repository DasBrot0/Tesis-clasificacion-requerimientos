#!/usr/bin/env python3
"""P8 (señales neuronales): CometKiwi + LaBSE. Se ejecuta en el entorno aislado
C:/Users/Franco/.venvs/cometkiwi/Scripts/python.exe

    python p8_neural_signals.py <in.csv(en,es)> <out.csv>

Salida: columnas cometkiwi y labse (coseno EN-ES; LaBSE vía transformers, pooler_output normalizado).
"""
import sys

import pandas as pd
import torch


def main(inp, out):
    df = pd.read_csv(inp, encoding="utf-8-sig")
    from comet import download_model, load_from_checkpoint
    model = load_from_checkpoint(download_model("Unbabel/wmt22-cometkiwi-da"))
    data = [{"src": a, "mt": b} for a, b in zip(df["en"], df["es"])]
    res = model.predict(data, batch_size=16, gpus=0)
    df["cometkiwi"] = res.scores

    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
    lab = AutoModel.from_pretrained("sentence-transformers/LaBSE").eval()

    def emb(texts):
        outs = []
        for i in range(0, len(texts), 32):
            b = tok(texts[i:i + 32], padding=True, truncation=True, max_length=256, return_tensors="pt")
            with torch.no_grad():
                e = lab(**b).pooler_output
            outs.append(torch.nn.functional.normalize(e, dim=1))
        return torch.cat(outs)

    a, b = emb(df["en"].tolist()), emb(df["es"].tolist())
    df["labse"] = (a * b).sum(dim=1).numpy()
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(df[["cometkiwi", "labse"]].describe())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
