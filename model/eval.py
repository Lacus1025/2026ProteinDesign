import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from esm_utils.esmc_embedding import ESM_embedding
from model.BrightnessRegressor import BrightnessRegressor

class EVAL:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.embedding = ESM_embedding()

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.model = BrightnessRegressor(1152)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict):
            self.model.load_state_dict(checkpoint)
        else:
            self.model = checkpoint

        self.model = self.model.to(device)
        self.model.eval()
        print(f"✓ 模型加载成功")

    def _process_embedding(self, emb):
        if torch.is_tensor(emb):
            emb = emb.detach().cpu().numpy()

        emb = np.squeeze(emb, axis=0)

        if len(emb.shape) == 2:
            emb = emb[1:-1]  # 剔除首尾token
            emb = emb.mean(axis=0)  # 平均池化得到 [1152]

        return emb.astype(np.float32)

    def predict(self, seq):
        with torch.no_grad():
            emb = self.embedding.embedding_sequence(seq)
            emb = self._process_embedding(emb)
            data = torch.from_numpy(emb).float().unsqueeze(0).to(self.device)
            output = self.model(data)
            return output.item()


if __name__ == "__main__":
    model_file = 'model_final.pth' if os.path.exists('model_final.pth') else 'model.pth'
    eval_model = EVAL(model_file)
    test_seq = 'MTALTEGAKLFEKEIPYITELEGDVEGMKFIIKGEGTGDATTGTIKAKYICTTGDLPVPWATILSSLSYGVFCFAKYPRHIADFFKSTQPDGYSQDRIISFDNDGQYDVKAKVTYENGTLYNRVTVKGTGFKSNGNILGMRVLYHSPPHAVYILPDRKNGGMKIEYNKAFDVMGGGHQMARHAQFNKPLGAWEEDYPLYHHLTVWTSFGKDPDDDETDHLTIVEVIKAVDLETYR'
    result = eval_model.predict(test_seq)
    print(f"序列: {test_seq}")
    print(f"预测亮度: {result:.4f}")
