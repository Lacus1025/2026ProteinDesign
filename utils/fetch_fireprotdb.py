import argparse
import json
import random
import sys
import time

import requests

BASE_URL = "https://loschmidt.chemi.muni.cz/fireprotdb"

TM_QUERY = json.dumps(
    {
        "tree": {
            "operator": "AND",
            "children": [{"variable": "TM", "operator": "IS_EMPTY", "value": False}],
        }
    }
)


def _extract_tm(measurements: list[dict]) -> float | None:
    for m in measurements:
        if m.get("type") == "TM" and m.get("numValue") is not None:
            return float(m["numValue"])
    return None


def fetch_tm_entries() -> list[dict]:
    params = {
        "query": TM_QUERY,
        "format": "jsonl",
    }

    resp = requests.get(
        f"{BASE_URL}/api/search",
        params=params,
        stream=True,
        timeout=(10, 120),
    )
    resp.raise_for_status()

    entries: list[dict] = []
    seen_ids: set[int] = set()

    for line in resp.iter_lines(decode_unicode=True):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        seq_data = entry.get("sequence")
        if seq_data is None:
            continue

        sid = seq_data.get("id")
        if sid is None or sid in seen_ids:
            continue
        seen_ids.add(sid)

        tm = _extract_tm(seq_data.get("experiment", {}).get("measurements", []))
        if tm is None:
            continue

        entries.append({"id": sid, "tm": tm})

    return entries


def fetch_sequence(seq_id: int) -> str | None:
    url = f"{BASE_URL}/api/sequences/{seq_id}/sequence"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.text.strip()
    except requests.RequestException:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 FireProtDB 随机获取蛋白序列和 Tm 值"
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=50,
        help="需要获取的蛋白条数 (default: 50)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="fireprotdb_sample.json",
        help="输出 JSON 文件路径 (default: fireprotdb_sample.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (default: 42)",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    print("正在从 FireProtDB 获取带 Tm 的蛋白条目...")
    entries = fetch_tm_entries()
    print(f"共找到 {len(entries)} 条带 Tm 的蛋白序列。")

    if len(entries) < args.num:
        print(
            f"警告: 可用条目 ({len(entries)}) 少于请求数量 ({args.num})，将获取全部。",
            file=sys.stderr,
        )

    sample = random.sample(entries, min(args.num, len(entries)))

    results: list[dict] = []
    for i, entry in enumerate(sample, 1):
        print(f"[{i}/{len(sample)}] 获取序列 ID={entry['id']} ...", end=" ")
        seq = fetch_sequence(entry["id"])
        if seq:
            results.append({"id": entry["id"], "sequence": seq, "tm": entry["tm"]})
            print(f"OK (len={len(seq)}, Tm={entry['tm']})")
        else:
            print("失败")
        time.sleep(0.1)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n已保存 {len(results)} 条蛋白到 {args.output}")


if __name__ == "__main__":
    main()
