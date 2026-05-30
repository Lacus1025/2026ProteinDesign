import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from utils.convert_gfp_data import get_json_sequence
from esm_utils.esmc_embedding import ESM_embedding


def export_dataset_json(output_path, batch=None, chunk_size=1000, shard_size=1000):
    df = pd.read_excel("./GFP_data.xlsx")

    data = get_json_sequence(df, batch)

    valid_data = [item for item in data if 225 <= len(item["sequence"]) <= 250]
    total = len(valid_data)
    print(f"有效序列数: {total}")

    embedding_model = ESM_embedding()

    embed_dim = 2560
    base = output_path.replace(".json", "").rstrip("_")
    num_shards = (total + shard_size - 1) // shard_size
    shard_list = []

    for shard_idx in range(num_shards):
        start = shard_idx * shard_size
        end = min(start + shard_size, total)
        shard_data = valid_data[start:end]
        shard_count = len(shard_data)

        shard_json_path = f"{base}_shard_{shard_idx:03d}.json"
        shard_npy_path = f"{base}_shard_{shard_idx:03d}_embeddings.npy"
        raw_path = shard_npy_path + ".raw"
        mmap = np.memmap(
            raw_path, dtype="float32", mode="w+", shape=(shard_count, embed_dim)
        )

        with open(shard_json_path, "w", encoding="utf-8") as f:
            f.write("[\n")
            for local_i, item in enumerate(
                tqdm(shard_data, desc=f"Shard {shard_idx}", leave=False)
            ):
                seq = item["sequence"]

                emb = embedding_model.embedding_sequence(seq)
                if torch.is_tensor(emb):
                    emb = emb.detach().cpu().numpy()
                emb = emb.mean(axis=0).astype(np.float32)
                mmap[local_i] = emb

                record = {
                    "index": item["index"],
                    "sequence": item["sequence"],
                    "type": item["type"],
                    "brightness": item["brightness"],
                }
                json_line = json.dumps(record, ensure_ascii=False)
                if local_i > 0:
                    f.write(",\n")
                f.write(f"  {json_line}")
            f.write("\n]")

        mmap.flush()
        del mmap

        raw = np.memmap(
            raw_path, dtype="float32", mode="r", shape=(shard_count, embed_dim)
        )
        np.save(shard_npy_path, np.array(raw))
        os.remove(raw_path)
        del raw

        shard_list.append(
            {
                "shard": shard_idx,
                "json": os.path.basename(shard_json_path),
                "npy": os.path.basename(shard_npy_path),
                "count": shard_count,
            }
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"  Shard {shard_idx}: {shard_count} records → {shard_json_path}")

    manifest = {
        "total_records": total,
        "num_shards": num_shards,
        "embed_dim": embed_dim,
        "shards": shard_list,
    }
    manifest_path = f"{base}_shards.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Exported {total} records across {num_shards} shards → {manifest_path}")


if __name__ == "__main__":
    export_dataset_json("gfp_dataset.json")
