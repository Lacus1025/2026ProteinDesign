from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import torch
import pandas as pd

from convert_gfp_data import get_json_sequence

class ESM_embedding():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ESMC.from_pretrained("esmc_300m").to(self.device)

    def _create_protein(self,sequence):
        return ESMProtein(sequence)

    def _encode_protein_tensor(self,protein):
        return self.model.encode(protein)

    def _get_embedding(self,protein_tensor):
        return self.model.logits(
            protein_tensor,
            LogitsConfig(sequence=True, return_embeddings=True)
        )

    def embedding_sequence(self, sequence):
        return self._get_embedding(self._encode_protein_tensor(self._create_protein(sequence)))

if __name__ == '__main__':
    df = pd.read_excel('./GFP_data.xlsx')
    embedding = ESM_embedding()
    data = get_json_sequence(df,5)
    for i in range(len(data)):
        sequence = data[i]["sequence"]
        result = embedding.embedding_sequence(sequence)
        print(result)
