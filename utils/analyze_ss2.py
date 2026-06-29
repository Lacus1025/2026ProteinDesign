import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass, field


SEPARATOR = "# PSIPRED VFORMAT"


@dataclass
class ProteinSSStats:
    protein_id: int
    total: int = 0
    num_C: int = 0
    num_H: int = 0
    num_E: int = 0

    @property
    def ratio_C(self) -> float:
        return self.num_C / self.total if self.total > 0 else 0.0

    @property
    def ratio_H(self) -> float:
        return self.num_H / self.total if self.total > 0 else 0.0

    @property
    def ratio_E(self) -> float:
        return self.num_E / self.total if self.total > 0 else 0.0


def parse_ss2(filepath: str) -> list[ProteinSSStats]:
    proteins: list[ProteinSSStats] = []
    current_lines: list[str] = []

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            if line.startswith(SEPARATOR):
                if current_lines:
                    stats = _process_block(current_lines, len(proteins) + 1)
                    if stats is not None:
                        proteins.append(stats)
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            stats = _process_block(current_lines, len(proteins) + 1)
            if stats is not None:
                proteins.append(stats)

    return proteins


def _process_block(lines: list[str], protein_id: int) -> ProteinSSStats | None:
    stats = ProteinSSStats(protein_id=protein_id)

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        ss = parts[2]
        if ss in ("C", "H", "E"):
            stats.total += 1
            if ss == "C":
                stats.num_C += 1
            elif ss == "H":
                stats.num_H += 1
            elif ss == "E":
                stats.num_E += 1

    if stats.total == 0:
        return None
    return stats


def print_table(proteins: list[ProteinSSStats]) -> None:
    header = f"{'Protein':>8}  {'Total':>6}  {'C':>6}  {'H':>6}  {'E':>6}  {'C%':>7}  {'H%':>7}  {'E%':>7}"
    print(header)
    print("-" * len(header))

    for p in proteins:
        print(
            f"{p.protein_id:>8}  {p.total:>6}  {p.num_C:>6}  {p.num_H:>6}  {p.num_E:>6}"
            f"  {p.ratio_C:>6.1%}  {p.ratio_H:>6.1%}  {p.ratio_E:>6.1%}"
        )


def print_extremes(proteins: list[ProteinSSStats]) -> None:
    if not proteins:
        return

    for ss_label, ratio_attr, count_attr in [
        ("C", "ratio_C", "num_C"),
        ("H", "ratio_H", "num_H"),
        ("E", "ratio_E", "num_E"),
    ]:
        best = max(proteins, key=lambda p: getattr(p, ratio_attr))
        worst = min(proteins, key=lambda p: getattr(p, ratio_attr))

        print(
            f"\n[{ss_label}] 最高比例: Protein {best.protein_id:>4}  {getattr(best, ratio_attr):.1%}"
            f"  (C={best.num_C}, H={best.num_H}, E={best.num_E}, total={best.total})"
        )
        print(
            f"[{ss_label}] 最低比例: Protein {worst.protein_id:>4}  {getattr(worst, ratio_attr):.1%}"
            f"  (C={worst.num_C}, H={worst.num_H}, E={worst.num_E}, total={worst.total})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="分析 ss2 文件中每条蛋白的二级结构 C/H/E 比例与数量"
    )
    parser.add_argument(
        "-i",
        "--input",
        default="fasta_output.ss2",
        help="输入的 ss2 文件路径 (default: fasta_output.ss2)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="只输出极值摘要，不输出全表",
    )
    args = parser.parse_args()

    proteins = parse_ss2(args.input)
    if not proteins:
        print("未找到任何蛋白数据。", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print_table(proteins)

    print(f"\n共解析 {len(proteins)} 条蛋白。")
    print_extremes(proteins)


if __name__ == "__main__":
    main()
