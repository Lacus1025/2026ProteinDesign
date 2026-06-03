import bisect
import datetime
import json
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.BrightnessRegressor import BrightnessRegressor


SEED = 114514
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "embed_dim": 12800,
    "learning_rate": 0.0005,
    "num_epochs": 500,
    "checkpoint_freq": 30,
    "early_stopping_patience": 40,
    "checkpoint_dir": "checkpoints",
    "log_file": "training_logs.json",
    "device": DEVICE.type,
}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def print_device_info() -> None:
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


class GFP_Dataset(Dataset):
    """Loads GFP embeddings and brightness labels from JSON + .npy shards."""

    def __init__(self, file: str) -> None:
        manifest_path = file.replace(".json", "_shards.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            self.data = []
            for shard in manifest["shards"]:
                shard_json = os.path.join(os.path.dirname(file), shard["json"])
                with open(shard_json, "r", encoding="utf-8") as f:
                    self.data.extend(json.load(f))
            self._shards_info = manifest["shards"]
            self._mmaps: list[np.ndarray] | None = None
            self._base_dir = os.path.dirname(file)
            self._shard_offsets = [0]
            for shard in self._shards_info:
                self._shard_offsets.append(self._shard_offsets[-1] + shard["count"])
        else:
            self.data = json.load(open(file))
            embeddings_path = file.replace(".json", "_embeddings.npy")
            self.embeddings = np.load(embeddings_path).astype(np.float32)
            self._shards_info = None

    def _init_mmaps(self) -> None:
        if self._mmaps is None and self._shards_info is not None:
            self._mmaps = []
            for shard in self._shards_info:
                shard_npy = os.path.join(self._base_dir, shard["npy"])
                self._mmaps.append(np.load(shard_npy, mmap_mode="r"))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._shards_info is not None:
            self._init_mmaps()
            shard_idx = bisect.bisect_right(self._shard_offsets, idx) - 1
            local_idx = idx - self._shard_offsets[shard_idx]
            embedding = self._mmaps[shard_idx][local_idx].astype(np.float32)
        else:
            embedding = self.embeddings[idx]
        embedding = torch.from_numpy(embedding)
        label = np.log1p(max(0, self.data[idx]["brightness"]))
        label = torch.tensor(label, dtype=torch.float32)
        return embedding, label


def save_checkpoint(
    state: dict,
    is_best: bool,
    checkpoint_dir: str = "checkpoints",
    filename: str = "checkpoint.pth.tar",
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth.tar")
        shutil.copyfile(checkpoint_path, best_path)
        print(f"  -> Saved best model to {best_path}")


def load_checkpoint(
    checkpoint_path: str, model: nn.Module, optimizer: torch.optim.Optimizer | None = None
) -> tuple[int, float]:
    if not os.path.exists(checkpoint_path):
        return 0, float("inf")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]
    best_loss = checkpoint.get("best_loss", float("inf"))
    print(f"Loaded checkpoint: epoch {start_epoch}, loss {best_loss:.6f}")
    return start_epoch, best_loss


def log_training_info(
    log_path: str,
    epoch: int,
    train_loss: float,
    val_loss: float | None = None,
    lr: float | None = None,
    is_checkpoint: bool = False,
) -> dict:
    logs: list[dict] = []
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
    entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "epoch": epoch,
        "train_loss": float(train_loss),
        "is_checkpoint": is_checkpoint,
    }
    if val_loss is not None:
        entry["val_loss"] = float(val_loss)
    if lr is not None:
        entry["learning_rate"] = float(lr)
    logs.append(entry)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
    return entry


def evaluate_model(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, float, list[float], list[float]]:
    """Returns (avg_loss, r2, predictions, true_values) with brightness restored via expm1."""
    model.eval()
    total_loss = 0.0
    predictions: list[float] = []
    true_values: list[float] = []
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * len(data)
            predictions.extend(float(np.expm1(p)) for p in outputs.cpu().numpy().flatten())
            true_values.extend(float(np.expm1(t)) for t in labels.cpu().numpy().flatten())
    avg_loss = total_loss / len(loader.dataset)
    r2 = r2_score(true_values, predictions)
    return avg_loss, r2, predictions, true_values


def evaluate_on_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_samples: int = 100,
) -> dict:
    """Evaluate on test set and print random sample predictions."""
    test_loss, test_r2, all_preds, all_trues = evaluate_model(model, test_loader, criterion, device)
    details = []
    for p, t in zip(all_preds, all_trues):
        rel_sq = ((p - t) / t) ** 2 if t != 0 else (p - t) ** 2
        details.append({"pred": p, "true": t, "err": p - t, "rel_sq": rel_sq})

    print(f"  -> Test Loss: {test_loss:.6f}, R2: {test_r2:.4f}")
    if details:
        indices = np.random.choice(len(details), min(num_samples, len(details)), replace=False)
        for i in indices:
            r = details[i]
            print(f"  predicted:{r['pred']:.4f}  labels:{r['true']:.4f}  error:{r['err']:.4f}  loss:{r['rel_sq']:.6f}")

    return {"loss": test_loss, "r2": test_r2, "details": details}


def build_results_entry(
    avg_loss: float,
    r2: float,
    best_val_loss: float,
    epoch: int,
    detailed_results: list[dict],
) -> dict:
    return {
        "evaluation_summary": {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_samples": 0,
            "mean_squared_error": round(avg_loss, 6),
            "r2_score": round(r2, 4),
            "best_val_loss_during_training": round(best_val_loss, 6),
            "final_epoch": epoch,
            "device": DEVICE.type if torch.cuda.is_available() else "cpu",
        },
        "detailed_predictions": detailed_results[:50],
        "training_config": CONFIG,
    }


def save_evaluation_results(results_file: str, entry: dict) -> None:
    results_list: list[dict] = []
    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as f:
            try:
                results_list = json.load(f)
                if not isinstance(results_list, list):
                    results_list = [results_list]
            except Exception:
                results_list = []
    results_list.append(entry)
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results_list, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {results_file} ({len(results_list)} runs)")


def make_checkpoint_state(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_loss: float,
    train_loss: float,
    val_loss: float,
) -> dict:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_loss": best_loss,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": CONFIG,
    }


def main() -> None:
    set_seed(SEED)
    print_device_info()

    dataset = GFP_Dataset("/data1/user/wuruiluo/protein_design/2026ProteinDesign/gfp_dataset.json")
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    num_workers = 4 if DEVICE.type == "cuda" else 0
    pin = DEVICE.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=num_workers, pin_memory=pin)

    print(f"Dataset: {len(dataset)} | Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    model = BrightnessRegressor(embed_dim=CONFIG["embed_dim"]).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15)

    checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], "checkpoint.pth.tar")
    start_epoch, best_val_loss = load_checkpoint(checkpoint_path, model, optimizer)

    num_epochs = CONFIG["num_epochs"]
    patience_counter = 0

    print(f"Training: epoch {start_epoch + 1} -> {num_epochs}")
    print(f"Checkpoint every {CONFIG['checkpoint_freq']} epochs, early stop patience {CONFIG['early_stopping_patience']}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0
        for data, labels in train_loader:
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            outputs = model(data)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_loss, val_r2, _, _ = evaluate_model(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch + 1}/{num_epochs}]  Train: {avg_train_loss:.6f}  Val: {val_loss:.6f}  R2: {val_r2:.4f}  LR: {current_lr:.2e}")

        log_training_info(CONFIG["log_file"], epoch + 1, avg_train_loss, val_loss, current_lr)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"  -> New best model! Val loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1

        if (epoch + 1) % CONFIG["checkpoint_freq"] == 0 or epoch == num_epochs - 1:
            ckpt_state = make_checkpoint_state(epoch + 1, model, optimizer, best_val_loss, avg_train_loss, val_loss)
            save_checkpoint(ckpt_state, is_best, CONFIG["checkpoint_dir"], f"checkpoint_epoch_{epoch + 1}.pth.tar")
            log_training_info(CONFIG["log_file"], epoch + 1, avg_train_loss, val_loss, current_lr, is_checkpoint=True)
            evaluate_on_test_set(model, test_loader, criterion, DEVICE)

        if patience_counter >= CONFIG["early_stopping_patience"]:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    final_path = "model_final.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "config": CONFIG,
    }, final_path)
    print(f"Final model saved to {final_path}")

    print("\n" + "=" * 50)
    print("Final Evaluation")
    print("=" * 50)

    best_path = os.path.join(CONFIG["checkpoint_dir"], "best_model.pth.tar")
    if os.path.exists(best_path):
        print(f"Loading best checkpoint: {best_path}")
        load_checkpoint(best_path, model)

    result = evaluate_on_test_set(model, test_loader, criterion, DEVICE, num_samples=100)
    entry = build_results_entry(result["loss"], result["r2"], best_val_loss, epoch + 1, result["details"])
    save_evaluation_results("evaluation_results.json", entry)

    print(f"\nFinal MSE: {result['loss']:.6f}  R2: {result['r2']:.4f}")

    if torch.cuda.is_available():
        print(f"\nGPU memory: allocated {torch.cuda.memory_allocated() / 1e9:.2f} GB, cached {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
