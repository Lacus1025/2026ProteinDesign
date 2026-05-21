import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.convert_gfp_data import get_json_sequence
from esm_utils.esmc_embedding import ESM_embedding


def export_dataset_json(output_path, batch=None):
    df = pd.read_excel("./GFP_data.xlsx")

    data = get_json_sequence(df, batch)

    embedding_model = ESM_embedding()
    all_embeddings = []

    serializable_data = []

    data = data[:10000]
    for item in tqdm(data, desc="Exporting dataset"):
        seq = item["sequence"]
        emb = embedding_model.embedding_sequence(seq)
        emb = emb.squeeze(0).flatten()
        all_embeddings.append(emb)

        serializable_data.append(
            {
                "index": item["index"],
                "sequence": item["sequence"],
                "type": item["type"],
                "brightness": item["brightness"],
            }
        )

    embeddings_array = np.stack(all_embeddings)
    embeddings_path = output_path.replace(".json", "_embeddings.npy")
    np.save(embeddings_path, embeddings_array)
    print(
        f"Exported {len(all_embeddings)} embeddings (shape: {embeddings_array.shape}) to {embeddings_path}"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_data, f, ensure_ascii=False, indent=2)

    print(f"Exported {len(serializable_data)} records to {output_path}")


if __name__ == "__main__":
    export_dataset_json("gfp_dataset.json")
