import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[1]
POSITIONS = {"subject": 0, "prep": 1, "prediction": 2}
COLORS = LinearSegmentedColormap.from_list("routes", ["#2878b5", "#8b4a9c", "#c83e3e"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/pythia-1b")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=int,
        default=[1000, 2000, 5000, 8000, 18000, 50000, 70000, 143000],
    )
    args = parser.parse_args()

    model_name = args.model.split("/")[-1]
    result_dir = ROOT / "results/experiment2_routing" / model_name
    figure_dir = ROOT / "figures"
    figure_dir.mkdir(exist_ok=True)
    figure, axes = plt.subplots(1, len(args.checkpoints), figsize=(2.5 * len(args.checkpoints), 5), sharey=True)

    for axis, checkpoint in zip(np.atleast_1d(axes), args.checkpoints):
        result = json.loads((result_dir / f"step{checkpoint}.json").read_text())
        max_relp = max(row["relp"] for row in result["neurons"])
        offsets = {}
        for row in result["neurons"]:
            key = (row["position"], row["layer"])
            offset = offsets.get(key, 0)
            offsets[key] = offset + 1
            x = POSITIONS[row["position"]] + (offset % 5 - 2) * 0.035
            color = "#aaaaaa" if not row["on_route"] else COLORS(row["route_share_l13"])
            axis.scatter(x, row["layer"], s=18 + 70 * row["relp"] / max_relp, color=color, edgecolor="none", zorder=3)

        l15 = max(0.0, result["head_sufficiency"]["L15H7_subject"])
        l13_subject = max(0.0, result["head_sufficiency"]["L13H7_subject"])
        l13_prep = max(0.0, result["head_sufficiency"]["L13H7_prep"])
        axis.plot([0, 2], [15, 15], color="#2878b5", linewidth=0.5 + 7 * l15, alpha=0.7)
        axis.plot([0, 2], [13, 13], color="#c83e3e", linewidth=0.5 + 7 * l13_subject, alpha=0.7)
        axis.plot([1, 2], [12.8, 13], color="#c83e3e", linewidth=0.5 + 7 * l13_prep, alpha=0.5)
        axis.set_title(f"{checkpoint // 1000}k")
        axis.set_xticks([0, 1, 2], ["subj", "prep", "pred"])
        axis.set_xlim(-0.35, 2.35)
        axis.set_ylim(0, 15.7)
        axis.grid(axis="y", linewidth=0.3, alpha=0.25)

    axes = np.atleast_1d(axes)
    axes[0].set_ylabel("layer")
    figure.tight_layout()
    output = figure_dir / "experiment2_joint_reorganization.pdf"
    figure.savefig(output, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
