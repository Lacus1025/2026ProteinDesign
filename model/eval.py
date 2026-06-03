import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from esm_utils.esmc_embedding import ESM_embedding
from model.BrightnessRegressor import BrightnessRegressor


class EVAL:
    def __init__(self, model_path, device="cuda:3"):
        # 自动降级：如果指定cuda但不可用，则回退到cpu
        if "cuda" in device and not torch.cuda.is_available():
            print("⚠️ CUDA不可用，将使用CPU")
            device = "cpu"
        self.device = device
        self.embedding = ESM_embedding(device=self.device)  # 如果ESM_embedding支持设备参数，可传入self.device

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.model = BrightnessRegressor()

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict):
            self.model.load_state_dict(checkpoint)
        else:
            self.model = checkpoint

        self.model = self.model.to(device)
        self.model.eval()
        print(f"✓ 模型加载成功，运行设备: {device}")

    def _process_embedding(self, emb):
        if torch.is_tensor(emb):
            emb = emb.detach().cpu().numpy()
        # embedding_sequence 返回 [250, 2560]，保留完整序列维度
        return emb.astype(np.float32)

    def predict(self, seq):
        with torch.no_grad():
            emb = self.embedding.embedding_sequence(seq)
            emb = self._process_embedding(emb)
            emb = emb.reshape(5, 50, 2560).mean(axis=1).reshape(-1)
            data = torch.from_numpy(emb).unsqueeze(0).to(self.device)
            output = self.model(data)
            return float(np.expm1(output.item()))


if __name__ == "__main__":
    # 自动选择GPU（如果可用）
    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    model_file = "../model_final.pth" if os.path.exists("../model_final.pth") else "../model.pth"
    eval_model = EVAL(model_file, device=device)
    test_seq = "MPLPATHDIHLHGSINGHEFDMVGGGKGDPNAGSLVTTAKSTKGALKFSPYLMIPHLGYGYYQYLPYPDGPSPFQVSMLEGSGYAVYRVFDFEDGGKLSTEFKYSYEGSHIKADMKLMGSGFPDDGPVMTSQIVDQDGCVSKKTYLNNNTIVDSFDWSYNLQNGKRYRARVSSHYIFDKPFSADLMKKQPVFVYRKCHVKATKTEVTLDEREKAFYELA"
    result = eval_model.predict(test_seq)
    print(f"序列: {test_seq}")
    print(f"预测亮度: {result:.4f}")