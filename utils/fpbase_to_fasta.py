import argparse
import json


def load_proteins(json_path: str) -> list[dict]:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def format_header(protein: dict) -> str:
    name = protein.get("name") or "-"
    slug = protein.get("slug") or "-"
    ex_nm = protein.get("ex_nm")
    em_nm = protein.get("em_nm")
    brightness = protein.get("brightness")

    ex_str = f"{ex_nm:.0f}" if ex_nm is not None else "-"
    em_str = f"{em_nm:.0f}" if em_nm is not None else "-"
    b_str = f"{brightness:.1f}" if brightness is not None else "-"

    return f"{name}|slug={slug}|ex={ex_str}|em={em_str}|brightness={b_str}"


def filter_proteins(
    proteins: list[dict],
    em_min: float,
    em_max: float,
) -> tuple[list[dict], dict]:
    kept = []
    skipped = {"no_sequence": 0, "no_em": 0, "out_of_range": 0}

    for p in proteins:
        seq = p.get("sequence")
        if not seq:
            skipped["no_sequence"] += 1
            continue

        em = p.get("em_nm")
        if em is None:
            skipped["no_em"] += 1
            continue

        if not (em_min <= em <= em_max):
            skipped["out_of_range"] += 1
            continue

        kept.append(p)

    return kept, skipped


def write_fasta(proteins: list[dict], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for p in proteins:
            header = format_header(p)
            seq = p["sequence"]
            f.write(f">{header}\n")
            f.write(f"{seq}\n")
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert fpbase JSON to FASTA, filtering by emission wavelength."
    )
    parser.add_argument(
        "-i",
        "--input",
        default="fpbase_proteins_complete.json",
        help="Input JSON file path (default: fpbase_proteins_complete.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="fpbase_proteins.fas",
        help="Output FASTA file path (default: fpbase_proteins.fas)",
    )
    parser.add_argument(
        "--em-min",
        type=float,
        default=500.0,
        help="Minimum emission wavelength in nm (default: 500)",
    )
    parser.add_argument(
        "--em-max",
        type=float,
        default=530.0,
        help="Maximum emission wavelength in nm (default: 530)",
    )
    args = parser.parse_args()

    all_proteins = load_proteins(args.input)
    print(f"总蛋白数: {len(all_proteins)}")

    filtered, skipped = filter_proteins(all_proteins, args.em_min, args.em_max)
    print(
        f"跳过: 无序列 {skipped['no_sequence']}, 无em {skipped['no_em']}, "
        f"波长超出[{args.em_min}-{args.em_max}] {skipped['out_of_range']}"
    )
    print(f"导出: {len(filtered)} 条")

    write_fasta(filtered, args.output)
    print(f"已保存到 {args.output}")


if __name__ == "__main__":
    main()
