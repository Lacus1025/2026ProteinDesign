import requests
import json
import csv
from typing import List, Dict

BASE_URL = "https://www.fpbase.org"
API_URL = f"{BASE_URL}/api/proteins/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
def fetch_all_proteins() -> List[Dict]:
    """通过 FPbase REST API 获取所有蛋白质数据（单次请求）"""
    params = {"format": "json"}
    response = requests.get(API_URL, params=params, headers=HEADERS, timeout=60)
    response.raise_for_status()
    return response.json()


def extract_protein_info(protein: Dict) -> Dict:
    """从 API 返回的原始数据中提取关键字段"""
    states = protein.get("states", [])
    default_state = states[0] if states else {}

    agg_map = {"m": "monomer", "d": "dimer", "t": "tetramer", "wd": "weak dimer"}
    agg_raw = protein.get("agg", "")
    state = agg_map.get(agg_raw, agg_raw) if agg_raw else None

    slug = protein.get("slug", "")
    detail_url = f"{BASE_URL}/protein/{slug}/" if slug else None

    return {
        "name": protein.get("name"),
        "slug": slug,
        "detail_url": detail_url,
        "sequence": protein.get("seq"),
        "ex_nm": default_state.get("ex_max"),
        "em_nm": default_state.get("em_max"),
        "ext_coeff": default_state.get("ext_coeff"),
        "qy": default_state.get("qy"),
        "brightness": default_state.get("brightness"),
        "pka": default_state.get("pka"),
        "maturation": default_state.get("maturation"),
        "lifetime": default_state.get("lifetime"),
        "state": state,
        "switch_type": protein.get("switch_type") or None,
        "pdb": protein.get("pdb", []),
        "doi": protein.get("doi"),
    }


def main():
    print("开始通过 FPbase API 获取蛋白质数据...")

    raw_proteins = fetch_all_proteins()
    print(f"共获取 {len(raw_proteins)} 个蛋白质")

    all_data = [extract_protein_info(p) for p in raw_proteins]

    # 保存为 JSON
    with open("fpbase_proteins_complete.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    print("已保存到 fpbase_proteins_complete.json")

    # 保存为 CSV
    if all_data:
        fieldnames = ["name", "slug", "sequence", "ex_nm", "em_nm",
                      "ext_coeff", "qy", "brightness", "pka",
                      "maturation", "lifetime", "state", "switch_type",
                      "pdb", "doi", "detail_url"]
        with open("fpbase_proteins_complete.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in all_data:
                writer.writerow(row)
        print("已保存到 fpbase_proteins_complete.csv")

    print("全部完成")


if __name__ == "__main__":
    main()
