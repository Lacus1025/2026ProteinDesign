# AGENTS.md — 2026ProteinDesign

## Project Overview

GFP brightness predictor using ESM protein embeddings + Conv1D regression. Part of a synthetic biology competition.

## Build / Lint / Test Commands

### Environment Setup
```bash
conda create -n esm python=3.11
conda activate esm
pip install -r requirements.txt
```

### Data Preparation
```bash
python utils/export_dataset_json.py
```
Outputs `gfp_dataset.json` + `gfp_dataset_embeddings.npy`.

### Training
```bash
python model/train.py
```
Reads `gfp_dataset_conv.json` + `_embeddings_conv.npy`. Saves checkpoints to `checkpoints/` and final model to `model_final.pth`. Logs to `training_logs.json`.

### Evaluation
```bash
python model/eval.py
```
Loads `model_final.pth` (or `model.pth`), runs a hardcoded test sequence, prints predicted brightness.

### Full Pipeline
```bash
bash run.sh              # conda activate esm → pip install → export_dataset_json → train.py
```

### Testing
**No test infrastructure exists.** No `tests/` directory, no `pytest`, no `unittest`.   
There is no single-test or single-package test command to use.

### Linting / Formatting
**No lint config files exist.** `ruff` cache is present (`.ruff_cache/`) but no `ruff.toml` or `pyproject.toml`. If adding linting, prefer:
```bash
ruff check .             # or: ruff check --fix .
ruff format .            # or: ruff format --check .
```

## Code Style Guidelines

### Imports
- Order: standard library → third-party → local modules
- One import per line (`import os` not `import os, sys`)
- Local module resolution uses `sys.path.insert(0, os.path.dirname(...))` pattern in entry-point scripts (train.py, eval.py, export_dataset_json.py)
- In library code, use relative or absolute imports without path hacks:
  ```python
  from esm_utils.esmc_embedding import ESM_embedding
  from model.BrightnessRegressor import BrightnessRegressor
  ```
- Remove unused imports before committing

### Formatting
- No trailing whitespace
- UTF-8 encoding for all files
- Prefer explicit `with open(...)` context managers over bare `open()`
- 4-space indentation (no tabs)
- Max line length: 120 characters (loose convention)

### Type Annotations
- Currently **not used** anywhere in the codebase
- New code SHOULD add type hints for all function signatures (Python 3.9+ style):
  ```python
  def predict(self, seq: str) -> float: ...
  def save_checkpoint(state: dict, is_best: bool, checkpoint_dir: str = "checkpoints") -> None: ...
  ```
- Prefer `list[X]` / `dict[str, X]` over `typing.List[X]` / `typing.Dict[str, X]` (Python 3.9+)

### Naming Conventions
| Category | Convention | Example |
|---|---|---|
| Classes | PascalCase | `BrightnessRegressor`, `GFP_Dataset`, `ESM_embedding`, `EVAL` |
| Functions/Methods | snake_case | `save_checkpoint`, `load_checkpoint`, `evaluate_model` |
| Constants | UPPER_CASE | `SEED`, `CONFIG`, `SEQUENCE_LEN`, `GFP_WT` |
| Variables | snake_case | `train_loader`, `best_val_loss`, `all_embeddings` |
| Private methods | `_leading_underscore` | `_process_embedding`, `_pad_embeddings` |

### Error Handling
- Use `torch.no_grad()` for all inference/eval paths
- Check `torch.cuda.is_available()` before GPU usage
- Validate checkpoint dict keys before `load_state_dict`
- Use `try/except` for JSON file reads (corrupt logs)
- Avoid bare `except:` — catch specific exceptions
- Prefer early returns / guard clauses over deep nesting
- Log errors via `print()` (no logging framework in use)

### Docstrings
- Optional but encouraged for public APIs
- Use one-line triple-double-quote docstrings describing return value
- Comments may be in Chinese or English (project is bilingual)

### PyTorch Best Practices
- Use `nn.Module` subclasses with `super().__init__()`
- Decorate inference methods with `@torch.no_grad()`
- Move model to device via `.to(device)` after construction
- Use `DataLoader` with `num_workers=4` on GPU, `0` on CPU
- Enable `pin_memory=True` when using CUDA
- Apply gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
- Set deterministic flags for reproducibility:
  ```python
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  ```
- Save checkpoints as `{"epoch": ..., "model_state_dict": ..., "optimizer_state_dict": ..., "best_loss": ...}`
- Use `.astype(np.float32)` for numpy embeddings before converting to tensors
- Target device is always `"cuda" if torch.cuda.is_available() else "cpu"`

### Package Structure
```
.
├── esm_utils/            # ESM model wrappers (embedding, generation, folding)
├── model/                # Model definition, training, evaluation
├── utils/                # Data conversion and processing scripts
├── main.py               # Entry point (currently minimal)
├── checkpoints/          # Trained model checkpoints
├── data files            # *.json, *.npy, *.xlsx at root level
└── requirements.txt      # Dependencies (NOTE: "panda" is a typo, should be "pandas")
```

### Git Conventions
- Default branch: `main`
- Secondary branch: `branch2`
- Remote: `git@github.com:Lacus1025/2026ProteinDesign.git`
- The `.gitignore` skips: `*.npy`, `*.pth`, `*.tar`, `__pycache__`, PDFs
- Avoid committing large binary files (embeddings, checkpoints)
