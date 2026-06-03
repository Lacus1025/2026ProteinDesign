import sys
import os
import json
import random
import time
import torch
import numpy as np
import gc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    "initial_sequence": "MPLPATHDIHLHGSINGHEFDMVGGGKGDPNAGSLVTTAKSTKGALKFSPYLMIPHLGYGYYQYLPYPDGPSPFQVSMLEGSGYAVYRVFDFEDGGKLSTEFKYSYEGSHIKADMKLMGSGFPDDGPVMTSQIVDQDGCVSKKTYLNNNTIVDSFDWSYNLQNGKRYRARVSSHYIFDKPFSADLMKKQPVFVYRKCHVKATKTEVTLDEREKAFYELA",
    "batch_size_per_parent": 200,
    "top_k_ratio": 0.1,
    "max_parents": 20,
    "drop_rate": 0.01,
    "temperature": 1.5,
    "num_rounds": 10,
    "model_path": "model_final.pth",
    "output_file": "pipeline_results.json",
    "seed": 42,
    "mask_token": "_",
    "min_sequence_length": 225,
    "max_sequence_length": 250,
    "device": "cuda:3",  # "auto" | "cuda" | "cpu"
}


def generate_batch(generator, masked_seq, batch_size, temperature):
    sequences = []
    for i in range(batch_size):
        temp = temperature * (0.8 + 0.4 * random.random())
        try:
            protein = generator.predict_sequence(masked_seq, temp)
            seq = protein.sequence
            if CONFIG["min_sequence_length"] <= len(seq) <= CONFIG["max_sequence_length"]:
                sequences.append(seq)
        except Exception as e:
            print(f"  Generation error (attempt {i+1}): {e}")
    return sequences


def evaluate_sequences(evaluator, sequences):
    results = []
    total = len(sequences)
    for idx, seq in enumerate(sequences):
        if (idx + 1) % max(1, total // 10) == 0:
            print(f"  Evaluating {idx + 1}/{total}...")
        try:
            brightness = evaluator.predict(seq)
            results.append({"sequence": seq, "brightness": round(brightness, 6)})
        except Exception as e:
            print(f"  Evaluation error at idx {idx}: {e}")
    return results


def select_top(results, ratio, max_count):
    if not results:
        return []
    results = sorted(results, key=lambda x: x["brightness"], reverse=True)
    n = max(1, min(max_count, int(len(results) * ratio)))
    return results[:n]


def drop_residues(sequence, drop_rate, mask_token="_"):
    seq_list = list(sequence)
    n_drop = max(1, int(len(seq_list) * drop_rate))
    indices = random.sample(range(len(seq_list)), n_drop)
    for i in indices:
        seq_list[i] = mask_token
    return "".join(seq_list)


def save_results(data, filepath, config):
    output = {"config": config, "rounds": data}
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def deduplicate_sequences(sequences):
    seen = set()
    result = []
    for s in sequences:
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result


def main():
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    device = CONFIG["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("Protein Design Pipeline")
    print("=" * 60)
    print(json.dumps(CONFIG, indent=2, ensure_ascii=False))

    from esm_utils.esm3_generate import ESM_generate
    from model.eval import EVAL

    all_rounds = []
    current_parents = [CONFIG["initial_sequence"]]
    global_top_sequences = []

    # Load ESM3 once, reuse across all rounds
    print("\nLoading ESM3 generator...")
    generator = ESM_generate(device=device)

    for round_idx in range(CONFIG["num_rounds"]):
        round_start = time.time()
        print(f"\n{'=' * 60}")
        print(f"Round {round_idx + 1}/{CONFIG['num_rounds']}")
        print(f"Parents: {len(current_parents)} | Target: {len(current_parents) * CONFIG['batch_size_per_parent']} sequences")
        print(f"{'=' * 60}")

        # Phase 1: Generation with ESM3
        print(f"\n[Phase 1] Generating {len(current_parents) * CONFIG['batch_size_per_parent']} sequences...")

        round_data = {
            "round": round_idx + 1,
            "parents": [],
            "generated": [],
            "top": [],
        }
        all_generated = []

        parent_count = len(current_parents)
        for i, parent_seq in enumerate(current_parents):
            masked_seq = drop_residues(
                parent_seq, CONFIG["drop_rate"], CONFIG["mask_token"]
            )

            print(f"  [{i + 1}/{parent_count}] Generating {CONFIG['batch_size_per_parent']} sequences...")
            parent_start = time.time()

            sequences = generate_batch(
                generator, masked_seq,
                CONFIG["batch_size_per_parent"],
                CONFIG["temperature"],
            )

            parent_elapsed = time.time() - parent_start
            round_data["parents"].append(
                {
                    "parent": parent_seq,
                    "masked": masked_seq if round_idx > 0 else parent_seq,
                    "generated_count": len(sequences),
                }
            )
            all_generated.extend(sequences)
            print(f"  [{i + 1}/{parent_count}] Generated {len(sequences)} valid seqs ({parent_elapsed:.1f}s)")

        valid_count = len(all_generated)
        if valid_count == 0:
            print(f"Warning: Round {round_idx + 1} generated 0 valid sequences. Expanding parent pool.")
            round_data["generated"] = []
            round_data["top"] = []
            all_rounds.append(round_data)
            save_results(all_rounds, CONFIG["output_file"], CONFIG)
            round_file = CONFIG["output_file"].replace(".json", f"_round{round_idx + 1}.json")
            with open(round_file, "w") as f:
                json.dump({"config": CONFIG, "rounds": [round_data]}, f, indent=2, ensure_ascii=False)
            continue

        all_generated = deduplicate_sequences(all_generated)
        print(f"Phase 1 done: {valid_count} generated, {len(all_generated)} unique")

        # Phase 2: Evaluation
        print(f"\n[Phase 2] Loading brightness evaluator...")
        evaluator = EVAL(CONFIG["model_path"], device=device)

        results = evaluate_sequences(evaluator, all_generated)
        round_data["generated"] = results

        del evaluator
        if device == "cuda":
            torch.cuda.empty_cache()

        # Phase 3: Select top and prepare next round
        top_results = select_top(results, CONFIG["top_k_ratio"], CONFIG["max_parents"])
        round_data["top"] = top_results

        if top_results:
            current_parents = [item["sequence"] for item in top_results]
            global_top_sequences.extend(top_results)
            print(f"\nPhase 3: Top {len(top_results)} selected:")
            for rank, item in enumerate(top_results, 1):
                seq = item["sequence"]
                short_seq = seq[:30] + "..." + seq[-30:] if len(seq) > 60 else seq
                print(f"  #{rank:>3}  brightness={item['brightness']:.4f}  {short_seq}")
        else:
            print("\nPhase 3: No sequences survived selection.")
            break

        all_rounds.append(round_data)
        save_results(all_rounds, CONFIG["output_file"], CONFIG)
        round_file = CONFIG["output_file"].replace(".json", f"_round{round_idx + 1}.json")
        with open(round_file, "w") as f:
            json.dump({"config": CONFIG, "rounds": [round_data]}, f, indent=2, ensure_ascii=False)

        elapsed = time.time() - round_start
        print(f"Round {round_idx + 1} done ({elapsed:.1f}s)\n")

    # Release ESM3 from GPU
    del generator
    gc.collect()
    if "cuda" in device:
        torch.cuda.empty_cache()

    # Final summary
    print(f"\n{'=' * 60}")
    print("Pipeline Complete")
    print(f"{'=' * 60}")

    global_top_sequences = sorted(
        global_top_sequences, key=lambda x: x["brightness"], reverse=True
    )

    final_output = {
        "config": CONFIG,
        "summary": {
            "total_rounds_completed": len(all_rounds),
            "best_sequence": global_top_sequences[0]["sequence"] if global_top_sequences else None,
            "best_brightness": global_top_sequences[0]["brightness"] if global_top_sequences else None,
            "top_10_overall": global_top_sequences[:10],
        },
        "rounds": all_rounds,
    }
    with open(CONFIG["output_file"], "w") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"\nBest brightness: {final_output['summary']['best_brightness']}")
    print(f"Results saved to {CONFIG['output_file']}")
    print(f"Total JSON size: {os.path.getsize(CONFIG['output_file']) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
