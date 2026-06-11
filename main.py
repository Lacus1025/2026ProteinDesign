import sys
import os
import json
import random
import time
import torch
import numpy as np
import gc

sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    "batch_size_per_parent": 200,
    "top_k_ratio": 0.1,
    "max_parents": 20,
    "drop_rate": 0.01,
    "temperature": 1.5,
    "num_rounds": 10,
    "model_path": "model_final.pth",
    "seed": 42,
    "mask_token": "_",
    "min_sequence_length": 225,
    "max_sequence_length": 250,
    "device": "cuda",
    "tm_config": "configs/S_esmc/model3.py",
    "tm_checkpoint": "results/S_esmc/seed-101/model3/epoch_best.pth",
}

LOCKED_POSITIONS = {
    # "avGFP": {1: "M", 64: "L", 65: "S", 66: "Y", 67: "G", 94: "Q",
            #   96: "R", 148: "H", 203: "T", 205: "S", 206: "A", 222: "E"},
    "sfGFP": {1: "M", 30: "R", 39: "N", 64: "L", 65: "T", 66: "Y",
              67: "G", 94: "Q", 96: "R", 99: "S", 105: "T", 145: "F",
              147: "P", 148: "H", 153: "T", 163: "A", 171: "V", 203: "T",
              205: "S", 206: "V", 222: "E"},
}


def read_reference_file(path):
    seqs = {}
    name = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                name = line[1:].strip()
            elif name and line and not line.startswith("#"):
                seqs[name] = seqs.get(name, "") + line
    return seqs


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


def evaluate_sequences(evaluator, tm_predictor, sequences):
    results = []
    total = len(sequences)
    for idx, seq in enumerate(sequences):
        if (idx + 1) % max(1, total // 10) == 0:
            print(f"  Evaluating {idx + 1}/{total}...")
        try:
            brightness = evaluator.predict(seq)
            tm = tm_predictor.predict(seq)
            results.append({
                "sequence": seq,
                "brightness": round(brightness, 6),
                "tm": round(tm, 4),
            })
        except Exception as e:
            print(f"  Evaluation error at idx {idx}: {e}")
    return results


def select_top(results, ratio, max_count):
    if not results:
        return []
    results = sorted(results, key=lambda x: x["composite_score"], reverse=True)
    n = max(1, min(max_count, int(len(results) * ratio)))
    return results[:n]


def drop_residues(sequence, drop_rate, mask_token="_", locked_positions=None, ss_pred=None):
    seq_list = list(sequence)
    maskable = []
    for i in range(len(seq_list)):
        if locked_positions and (i + 1) in locked_positions:
            continue
        if ss_pred is not None and ss_pred[i] != "C":
            continue
        maskable.append(i)
    if not maskable:
        return sequence
    n_drop = max(1, min(len(maskable), int(len(seq_list) * drop_rate)))
    indices = random.sample(maskable, n_drop)
    for i in indices:
        seq_list[i] = mask_token
    return "".join(seq_list)


def save_results(data, filepath, config):
    output = {"config": config, "rounds": data}
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


INVALID_AA = set("BJOUXZ")

def deduplicate_sequences(sequences):
    seen = set()
    result = []
    for s in sequences:
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result


def load_exclusion_list(path="Exclusion_List.csv"):
    excluded = set()
    with open(path) as f:
        next(f)
        for line in f:
            seq = line.strip()
            if seq:
                excluded.add(seq)
    return excluded


def filter_valid_sequences(sequences, excluded_set=None):
    result = []
    for s in sequences:
        if any(aa in INVALID_AA for aa in s):
            continue
        if excluded_set and s in excluded_set:
            continue
        result.append(s)
    return result


def compute_composite_scores(results, tm_wt):
    if not results:
        return
    b_arr = np.array([r["brightness"] for r in results])
    tm_arr = np.array([r["tm"] for r in results])
    dtm_arr = tm_arr - tm_wt

    b_min, b_max = b_arr.min(), b_arr.max()
    dtm_min, dtm_max = dtm_arr.min(), dtm_arr.max()

    b_norm = np.full_like(b_arr, 0.5) if b_max == b_min else (b_arr - b_min) / (b_max - b_min)
    dtm_norm = np.full_like(dtm_arr, 0.5) if dtm_max == dtm_min else (dtm_arr - dtm_min) / (dtm_max - dtm_min)

    composite = b_norm * dtm_norm
    for i, r in enumerate(results):
        r["composite_score"] = round(float(composite[i]), 6)
        r["delta_tm"] = round(float(dtm_arr[i]), 4)


def run_pipeline(substrate_type, initial_sequence, config, s4pred, generator):
    from model.eval import EVAL
    from esm_utils.tm_predictor import TM_PREDICTOR

    output_file = f"pipeline_results_{substrate_type}.json"
    locked_positions = LOCKED_POSITIONS[substrate_type]
    device = config["device"]

    random.seed(config["seed"])
    np.random.seed(config["seed"])

    excluded_set = load_exclusion_list()

    print("\n" + "=" * 60)
    print(f"Pipeline: {substrate_type}")
    print(f"Initial sequence length: {len(initial_sequence)}")
    print(f"Locked positions: {len(locked_positions)}")
    print(f"Exclusion list loaded: {len(excluded_set)} sequences")
    print("=" * 60)

    print("\nLoading Tm predictor...")
    tm_predictor = TM_PREDICTOR(device=device, config_rel=config["tm_config"],
                                 checkpoint_rel=config["tm_checkpoint"])
    tm_wt = tm_predictor.predict(initial_sequence)
    print(f"WT Tm ({substrate_type}): {tm_wt:.2f}")

    all_rounds = []
    current_parents = [initial_sequence]
    global_top_sequences = []

    for round_idx in range(config["num_rounds"]):
        round_start = time.time()
        print(f"\n{'=' * 60}")
        print(f"[{substrate_type}] Round {round_idx + 1}/{config['num_rounds']}")
        print(f"Parents: {len(current_parents)} | "
              f"Target: {len(current_parents) * config['batch_size_per_parent']}")
        print(f"{'=' * 60}")

        print(f"\n[Phase 1] Loading evaluators...")
        evaluator = EVAL(config["model_path"], device=device)
        print(f"\n[Phase 2] Generating and evaluating...")

        round_data = {
            "round": round_idx + 1,
            "parents": [],
            "generated": [],
            "top": [],
        }
        all_results = []
        total_generated = 0

        parent_count = len(current_parents)
        for i, parent_seq in enumerate(current_parents):
            ss_pred = s4pred.predict(parent_seq)

            masked_seq = drop_residues(
                parent_seq, config["drop_rate"], config["mask_token"],
                locked_positions=locked_positions, ss_pred=ss_pred,
            )

            print(f"  [{i + 1}/{parent_count}] Generating "
                  f"{config['batch_size_per_parent']} sequences...")
            parent_start = time.time()

            sequences = generate_batch(
                generator, masked_seq,
                config["batch_size_per_parent"],
                config["temperature"],
            )

            n_raw = len(sequences)
            sequences = deduplicate_sequences(sequences)
            n_unique = len(sequences)
            sequences = filter_valid_sequences(sequences, excluded_set)
            n_valid = len(sequences)
            total_generated += n_valid

            parent_elapsed = time.time() - parent_start
            print(f"  [{i + 1}/{parent_count}] Generated {n_raw} raw, "
                  f"{n_unique} unique, {n_valid} valid ({parent_elapsed:.1f}s)")

            if n_valid == 0:
                round_data["parents"].append({
                    "parent": parent_seq,
                    "masked": masked_seq if round_idx > 0 else parent_seq,
                    "generated_count": 0,
                })
                continue

            batch_results = evaluate_sequences(evaluator, tm_predictor, sequences)
            compute_composite_scores(batch_results, tm_wt)

            round_data["parents"].append({
                "parent": parent_seq,
                "masked": masked_seq if round_idx > 0 else parent_seq,
                "generated_count": len(batch_results),
            })
            all_results.extend(batch_results)

        del evaluator
        if "cuda" in device:
            torch.cuda.empty_cache()

        if total_generated == 0:
            print(f"Warning: Round {round_idx + 1} generated 0 valid sequences.")
            round_data["generated"] = []
            round_data["top"] = []
            all_rounds.append(round_data)
            save_results(all_rounds, output_file, config)
            round_file = output_file.replace(".json", f"_round{round_idx + 1}.json")
            with open(round_file, "w") as f:
                json.dump({"config": config, "rounds": [round_data]}, f,
                          indent=2, ensure_ascii=False)
            continue

        round_data["generated"] = all_results
        print(f"Phase 2 done: {total_generated} valid seqs evaluated")

        top_results = select_top(all_results, config["top_k_ratio"], config["max_parents"])
        round_data["top"] = top_results

        if top_results:
            current_parents = [item["sequence"] for item in top_results]
            global_top_sequences.extend(top_results)
            print(f"\nPhase 3: Top {len(top_results)} selected:")
            for rank, item in enumerate(top_results, 1):
                seq = item["sequence"]
                short_seq = seq[:30] + "..." + seq[-30:] if len(seq) > 60 else seq
                print(f"  #{rank:>3}  score={item['composite_score']:.4f}  "
                      f"b={item['brightness']:.4f}  "
                      f"dTm={item.get('delta_tm', 0):+.2f}  {short_seq}")
        else:
            print("\nPhase 3: No sequences survived selection.")
            break

        all_rounds.append(round_data)
        save_results(all_rounds, output_file, config)
        round_file = output_file.replace(".json", f"_round{round_idx + 1}.json")
        with open(round_file, "w") as f:
            json.dump({"config": config, "rounds": [round_data]}, f,
                      indent=2, ensure_ascii=False)

        elapsed = time.time() - round_start
        print(f"Round {round_idx + 1} done ({elapsed:.1f}s)\n")

    del tm_predictor
    if "cuda" in device:
        torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print(f"[{substrate_type}] Pipeline Complete")
    print(f"{'=' * 60}")

    global_top_sequences = sorted(
        global_top_sequences, key=lambda x: x["composite_score"], reverse=True
    )

    final_output = {
        "config": config,
        "substrate": substrate_type,
        "summary": {
            "total_rounds_completed": len(all_rounds),
            "best_sequence": global_top_sequences[0]["sequence"]
                             if global_top_sequences else None,
            "best_brightness": global_top_sequences[0]["brightness"]
                               if global_top_sequences else None,
            "best_composite_score": global_top_sequences[0]["composite_score"]
                                   if global_top_sequences else None,
            "tm_wt": tm_wt,
            "top_10_overall": global_top_sequences[:10],
        },
        "rounds": all_rounds,
    }
    with open(output_file, "w") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"\nBest composite score: {final_output['summary']['best_composite_score']}")
    print(f"Best brightness: {final_output['summary']['best_brightness']}")
    print(f"Results saved to {output_file}")
    if os.path.exists(output_file):
        print(f"Total JSON size: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")

    return final_output


def main():
    device = CONFIG["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ref_path = "AAseqs of 5 GFP proteins_20260511.txt"
    ref_seqs = read_reference_file(ref_path)

    # av_ref = ref_seqs.get("avGFP", "")
    sf_ref = ref_seqs.get("sfGFP", "")

    # if not av_ref or not sf_ref:
        # raise ValueError(f"Cannot find avGFP/sfGFP in {ref_path}")

    sf_seq = sf_ref[:146] + "P" + sf_ref[147:]

    # print(f"avGFP reference: {len(av_ref)} aa (matches locked positions)")
    print(f"sfGFP reference: {len(sf_ref)} aa (S147P applied)")

    print("\nLoading S4PRED secondary structure predictor...")
    from esm_utils.s4pred_wrapper import S4PredWrapper
    s4pred_device = "cuda" if torch.cuda.is_available() and "cuda" in device else "cpu"
    s4pred = S4PredWrapper(device=s4pred_device)

    print("\nLoading ESM3 generator...")
    from esm_utils.esm3_generate import ESM_generate
    generator = ESM_generate(device=device)

    run_pipeline("sfGFP", sf_seq, CONFIG, s4pred, generator)

    del generator
    gc.collect()
    if "cuda" in device:
        torch.cuda.empty_cache()

    print("\nAll pipelines complete!")


if __name__ == "__main__":
    main()
