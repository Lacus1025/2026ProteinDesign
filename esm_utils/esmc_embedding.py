import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import torch
import pandas as pd

from utils.convert_gfp_data import get_json_sequence

SEQUENCE_LEN = 500

class ESM_embedding():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ESMC.from_pretrained("esmc_600m").to(self.device)
        self.sequence_len = SEQUENCE_LEN

    def _create_protein(self,sequence):
        return ESMProtein(sequence=sequence)

    def _encode_protein_tensor(self,protein):
        return self.model.encode(protein)

    def _get_embedding(self,protein_tensor):
        return self.model.logits(
            protein_tensor,
            LogitsConfig(sequence=True, return_embeddings=True)
        )

    def _pad_embeddings(self, embeddings, target_length):
        embeddings = embeddings.squeeze(0)
        current_length = embeddings.shape[0]
        embedding_dim = embeddings.shape[1]

        if current_length >= target_length:
            padded = embeddings[:target_length, :]
            print(f"  Truncated from {current_length} to {target_length}")
        else:
            padding_size = target_length - current_length
            padding = torch.zeros(padding_size, embedding_dim, device=embeddings.device)
            padded = torch.cat([embeddings, padding], dim=0)
            print(f"  Padded from {current_length} to {target_length} (added {padding_size} zeros)")
        padded = padded.unsqueeze(0)
        return padded

    def embedding_sequence(self, sequence):
        """返回序列的 embeddings 张量"""
        protein = self._create_protein(sequence)
        encoded = self._encode_protein_tensor(protein)
        result = self._get_embedding(encoded)

        if hasattr(result, 'embeddings'):
            embeddings = result.embeddings
        elif hasattr(result, 'logits') and hasattr(result.logits, 'embeddings'):
            embeddings = result.logits.embeddings
        else:
            print(f"Result structure: {dir(result)}")
            embeddings = None
        padded_embeddings = self._pad_embeddings(embeddings, SEQUENCE_LEN)
        if torch.is_tensor(padded_embeddings):
            return padded_embeddings.cpu().detach().numpy()
        return padded_embeddings

if __name__ == '__main__':
    df = pd.read_excel('./GFP_data.xlsx')
    embedding = ESM_embedding()
    data = get_json_sequence(df,1)
    for i in range(len(data)):
        sequence = data[i]["sequence"]
        result = embedding.embedding_sequence(sequence)
        print(result.shape)
