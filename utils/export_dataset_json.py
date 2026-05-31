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


def _write_shard(shard_idx, records, embeddings, base, embed_dim):
    shard_count = len(records)
    shard_json_path = f"{base}_shard_{shard_idx:03d}.json"
    shard_npy_path = f"{base}_shard_{shard_idx:03d}_embeddings.npy"
    raw_path = shard_npy_path + ".raw"

    mmap = np.memmap(
        raw_path, dtype="float32", mode="w+", shape=(shard_count, embed_dim)
    )
    stacked = np.stack(embeddings, axis=0)
    mmap[:] = stacked
    mmap.flush()
    del mmap

    with open(shard_json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    raw = np.memmap(
        raw_path, dtype="float32", mode="r", shape=(shard_count, embed_dim)
    )
    np.save(shard_npy_path, np.array(raw))
    os.remove(raw_path)
    del raw

    print(f"  Shard {shard_idx}: {shard_count} records → {shard_json_path}")

    return {
        "shard": shard_idx,
        "json": os.path.basename(shard_json_path),
        "npy": os.path.basename(shard_npy_path),
        "count": shard_count,
    }


def export_dataset_json(output_path, batch=None, chunk_size=1000, shard_size=1000):
    df = pd.read_excel("./GFP_data.xlsx")

    data = get_json_sequence(df, batch)
    print(f"原始序列数: {len(data)}")

    embedding_model = ESM_embedding()

    embed_dim = 2560
    base = output_path.replace(".json", "").rstrip("_")

    total = 0
    shard_idx = 0
    shard_list = []
    records_buffer = []
    embeddings_buffer = []

    total_chunks = (len(data) + chunk_size - 1) // chunk_size

    for chunk_start in tqdm(range(0, len(data), chunk_size), total=total_chunks, desc="Processing chunks"):
        chunk_end = min(chunk_start + chunk_size, len(data))
        chunk = data[chunk_start:chunk_end]

        for item in chunk:
            seq = item["sequence"]
            if not (225 <= len(seq) <= 250):
                continue

            emb = embedding_model.embedding_sequence(seq)
            if torch.is_tensor(emb):
                emb = emb.detach().cpu().numpy()
            emb = emb.mean(axis=0).astype(np.float32)

            records_buffer.append({
                "index": item["index"],
                "sequence": seq,
                "type": item["type"],
                "brightness": item["brightness"],
            })
            embeddings_buffer.append(emb)
            total += 1

            if len(records_buffer) >= shard_size:
                shard_list.append(_write_shard(
                    shard_idx, records_buffer, embeddings_buffer, base, embed_dim
                ))
                shard_idx += 1
                records_buffer = []
                embeddings_buffer = []
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if records_buffer:
        shard_list.append(_write_shard(
            shard_idx, records_buffer, embeddings_buffer, base, embed_dim
        ))
        shard_idx += 1
        records_buffer = []
        embeddings_buffer = []

    num_shards = len(shard_list)
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
