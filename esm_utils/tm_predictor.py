import sys
import os
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESA_DEFAULT = os.path.join(BASE_DIR, "ProCeSa", "procesa")


class TM_PREDICTOR:
    def __init__(self, device="cuda", procesa_dir=None,
                 config_rel="configs/S_esmc/model3.py",
                 checkpoint_rel="results/S_esmc/seed-101/model3/epoch_best.pth"):
        if "cuda" in device and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = "cpu"
        self.device = device

        if procesa_dir is None:
            procesa_dir = PROCESA_DEFAULT
        procesa_dir = os.path.abspath(procesa_dir)

        if procesa_dir not in sys.path:
            sys.path.insert(0, procesa_dir)

        from build_dgl_graph import normalize_adj
        from FLIP.baselines.esm_next.models.esmc import ESMC
        from utils import Config
        from models import build_model
        from utils_flip_self import load_ckpt

        self._normalize_adj = normalize_adj

        cfg = Config.fromfile(os.path.join(procesa_dir, config_rel))
        self.model = build_model(cfg.model)
        self.model, _ = load_ckpt(self.model, os.path.join(procesa_dir, checkpoint_rel))
        self.model.eval().to(self.device)

        self.encoder = ESMC.from_pretrained('esmc_600m').eval().to(self.device)

    @torch.no_grad()
    def predict(self, seq: str) -> float:
        import dgl

        seq = seq[:800]
        ids = self.encoder.tokenizer.encode(seq)
        toks = torch.tensor([ids], device=self.device)
        seq_id = torch.zeros_like(toks, dtype=torch.bool)

        _, embedding, _, attention = self.encoder(sequence_tokens=toks, sequence_id=seq_id)
        embedding = embedding[0, 1:len(seq) + 1].float().cpu().numpy()
        attention = attention[0, :len(seq), :len(seq)].float().cpu().numpy()

        src, dst = np.nonzero(attention)
        edge_feats = self._normalize_adj(attention)
        edge_feats = edge_feats[np.nonzero(edge_feats)]
        graph = dgl.graph((src, dst), num_nodes=embedding.shape[0]).to(self.device)
        graph.ndata['x'] = torch.from_numpy(embedding).float().to(self.device)
        graph.edata['x'] = torch.from_numpy(edge_feats).float().to(self.device)

        return float(self.model.forward_test(graph).item())
