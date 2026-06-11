#!/usr/bin/env python3
"""Predict thermostability (Tm) and brightness for GFP sequences using ProCeSa."""

import sys, os
import json
import numpy as np
import torch
import dgl
import openpyxl

PROCESA = os.path.join(os.path.dirname(__file__), '../ProCeSa/procesa')
sys.path.insert(0, PROCESA)

from build_dgl_graph import normalize_adj
from FLIP.baselines.esm_next.models.esmc import ESMC
from utils import Config
from models import build_model
from utils_flip_self import load_ckpt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.eval import EVAL

CONFIG = os.path.join(PROCESA, 'configs/S_esmc/model3.py')
CKPT = os.path.join(PROCESA, 'results/S_esmc/seed-101/model3/epoch_best.pth')
BRIGHTNESS_MODEL = os.path.join(os.path.dirname(__file__), 'model_final.pth')
GFP_DATA = os.path.join(os.path.dirname(__file__), 'GFP_data.xlsx')
SEQ_FILE = os.path.join(os.path.dirname(__file__), 'AAseqs of 5 GFP proteins_20260511.txt')
PIPELINE_AV = os.path.join(os.path.dirname(__file__), 'pipeline_results_avGFP.json')
PIPELINE_SF = os.path.join(os.path.dirname(__file__), 'pipeline_results_sfGFP.json')


def read_gfp_sequences(path):
    seqs = {}
    name, seq = None, []
    for line in open(path):
        line = line.strip()
        if line.startswith('>'):
            if name:
                seqs[name] = ''.join(seq)
            name = line[1:].split()[0]
            seq = []
        elif line and not line.startswith('#'):
            seq.append(line)
    if name:
        seqs[name] = ''.join(seq)
    return seqs


def encode(encoder, seq, device):
    seq = seq[:800]
    ids = encoder.tokenizer.encode(seq)
    toks = torch.tensor([ids], device=device)
    seq_id = torch.zeros_like(toks, dtype=torch.bool)
    with torch.no_grad():
        _, embedding, _, attention = encoder(sequence_tokens=toks, sequence_id=seq_id)
    embedding = embedding[0, 1:len(seq)+1].float().cpu().numpy()
    attention = attention[0, :len(seq), :len(seq)].float().cpu().numpy()
    return embedding, attention


def build_graph(embedding, attention, device):
    src, dst = np.nonzero(attention)
    edge_feats = normalize_adj(attention)
    edge_feats = edge_feats[np.nonzero(edge_feats)]
    graph = dgl.graph((src, dst), num_nodes=embedding.shape[0]).to(device)
    graph.ndata['x'] = torch.from_numpy(embedding).float().to(device)
    graph.edata['x'] = torch.from_numpy(edge_feats).float().to(device)
    return graph


def load_exclusion_list(path="Exclusion_List.csv"):
    excluded = set()
    with open(path) as f:
        next(f)
        for line in f:
            seq = line.strip()
            if seq:
                excluded.add(seq)
    return excluded


def collect_sequences():
    """Collect all sequences with metadata. Returns list of dicts
    with keys: source, sequence, extra (dict for brightness/year etc)."""
    items = []

    # 1. Parent sequences
    seqs = read_gfp_sequences(SEQ_FILE)
    items.append({'source': 'avGFP_parent', 'sequence': seqs.get('avGFP', ''), 'extra': {}})
    items.append({'source': 'sfGFP_parent', 'sequence': seqs.get('sfGFP', ''), 'extra': {}})

    # 2. Pipeline top10
    for pipe_file, prefix in [(PIPELINE_AV, 'avGFP_top'), (PIPELINE_SF, 'sfGFP_top')]:
        with open(pipe_file) as f:
            data = json.load(f)
        for i, item in enumerate(data['summary']['top_10_overall'], 1):
            items.append({
                'source': f'{prefix}{i}',
                'sequence': item['sequence'],
                'extra': {'brightness': item['brightness']},
            })

    # 3. beforetopseqs
    wb = openpyxl.load_workbook(GFP_DATA, data_only=True)
    ws = wb['beforetopseqs']
    for row in ws.iter_rows(min_row=2, values_only=True):
        year, seq = row
        if seq:
            items.append({'source': None, 'sequence': seq, 'extra': {'year': year}})
    wb.close()

    return items


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # --- Load ProCeSa model ---
    cfg = Config.fromfile(CONFIG)
    model = build_model(cfg.model)
    model, _ = load_ckpt(model, CKPT)
    model.eval().to(device)
    print(f'ProCeSa model loaded')

    # --- Load ESMC encoder ---
    encoder = ESMC.from_pretrained('esmc_600m').eval().to(device)
    print('ESMC encoder loaded')

    # --- Load brightness predictor ---
    brightness_model = EVAL(BRIGHTNESS_MODEL, device=device)
    print('Brightness model loaded')

    # --- Load exclusion list ---
    excluded_set = load_exclusion_list()
    print(f'Exclusion list loaded: {len(excluded_set)} sequences')

    # --- Collect sequences ---
    raw_items = collect_sequences()
    n_before = len(raw_items)
    raw_items = [item for item in raw_items
                 if (item['source'] and 'parent' in item['source']) or item['sequence'] not in excluded_set]
    n_excluded = n_before - len(raw_items)
    print(f'Total sequences: {len(raw_items)} ({n_excluded} excluded)')

    # --- Predict ---
    results = []
    for i, item in enumerate(raw_items, 1):
        source = item['source'] or f'beforetopseqs_{i}'
        seq = item['sequence']
        extra = dict(item['extra'])

        print(f'  [{i}/{len(raw_items)}] {source} ({len(seq)} aa)...', end=' ', flush=True)

        emb, attn = encode(encoder, seq, device)
        graph = build_graph(emb, attn, device)
        tm = model.forward_test(graph).item()

        extra['brightness'] = round(brightness_model.predict(seq), 6)

        results.append({
            'index': i,
            'source': source,
            'sequence': seq,
            'length': len(seq),
            'Tm_pred': round(tm, 2),
            **extra,
        })
        print(f'Tm={tm:.2f}  brightness={extra["brightness"]:.4f}')

    out_path = os.path.join(os.path.dirname(__file__), 'tm_predictions.json')
    with open(out_path, 'w') as f:
        json.dump({'model': 'ProCeSa S_esmc/model3', 'total': len(results), 'results': results}, f, indent=2)
    print(f'\nSaved to {out_path}')


if __name__ == '__main__':
    main()
