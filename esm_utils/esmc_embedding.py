import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

SEQUENCE_LEN = 250


class ESM_embedding:
    def __init__(self, device="cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            print("⚠️ CUDA不可用，将使用CPU")
            device = "cpu"
        self.device = device
        self.model = AutoModelForMaskedLM.from_pretrained("Biohub/ESMC-6B",).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained("Biohub/ESMC-6B")

    @staticmethod
    def _pad_embeddings(embeddings, target_length):
        current_length = embeddings.shape[0]
        embedding_dim = embeddings.shape[1]

        if current_length >= target_length:
            return embeddings[:target_length, :]
        else:
            padding_size = target_length - current_length
            padding = torch.zeros(padding_size, embedding_dim, device=embeddings.device)
            return torch.cat([embeddings, padding], dim=0)

    @torch.no_grad()
    def embedding_sequence(self, sequence):
        """返回固定长度的 per-token Embedding 张量 [250, 2560]（无 pooling）"""
        inputs = self.tokenizer(sequence, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model(**inputs, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1]

        embeddings = embeddings.squeeze(0)
        embeddings = self._pad_embeddings(embeddings, SEQUENCE_LEN)

        return embeddings.cpu().float().numpy()
