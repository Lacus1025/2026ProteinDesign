import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig

# TEMPERATURE = 1
NUM_STEP = 8

login()

class ESM_generate:
    def __init__(self):
        self.model: ESM3InferenceClient = ESM3.from_pretrained("esm3-open").to("cuda") # or "cpu"

    def predict_sequence(self,raw_sequence,temp):
        raw_sequence = ESMProtein(sequence=raw_sequence)
        return self.model.generate(raw_sequence,GenerationConfig(track="sequence", num_steps=NUM_STEP, temperature=temp))

if __name__ == '__main__':
    esm3 = ESM_generate()
    print(esm3.predict_sequence('____AAAA_____BBB'))
