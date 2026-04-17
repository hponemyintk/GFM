import os
import matplotlib.pyplot as plt
import numpy as np
from relbench.datasets import get_dataset
from relbench.tasks import get_task
from relbench.base import TaskType

dataset_task_dict = {
    "rel-f1": ["driver-position", "driver-dnf", "driver-top3"],
    "rel-avito": ["ad-ctr", "user-clicks", "user-visits"],
    "rel-event": ["user-attendance", "user-repeat", "user-ignore"],
    "rel-trial": ["study-adverse", "study-outcome", "site-success"],
    "rel-amazon": ["user-ltv", "item-ltv", "user-churn", "item-churn"],
    "rel-stack": ["post-votes", "user-engagement", "user-badge"],
    "rel-hm": ["item-sales", "user-churn"],
}

splits = ["train", "val", "test"]
split_colors = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}

outdir = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(outdir, exist_ok=True)

for dataset_name, task_names in dataset_task_dict.items():
    for task_name in task_names:
        print(f"Processing {dataset_name}/{task_name}...")
        try:
            task = get_task(dataset_name, task_name, download=True)
        except Exception as e:
            print(f"  FAILED to load task: {e}")
            continue

        is_classification = task.task_type in (
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION,
        )

        split_labels = {}
        for split in splits:
            try:
                table = task.get_table(split, mask_input_cols=False)
                df = table.df
                if task.target_col in df.columns:
                    split_labels[split] = df[task.target_col].dropna()
                else:
                    print(f"  {split}: labels hidden ({len(df)} rows)")
            except Exception as e:
                print(f"  {split} error: {e}")

        if not split_labels:
            print("  No labels available, skipping.")
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        tag = "classification" if is_classification else "regression"
        fig.suptitle(f"{dataset_name} / {task_name} ({tag})", fontsize=14, fontweight="bold")

        if is_classification:
            all_classes = sorted(set().union(*(labels.unique() for labels in split_labels.values())))
            n_classes = len(all_classes)
            n_splits = len(split_labels)
            bar_width = 0.8 / n_splits
            x = np.arange(n_classes)

            for i, (split, labels) in enumerate(split_labels.items()):
                counts = labels.value_counts()
                raw = np.array([counts.get(c, 0) for c in all_classes])
                heights = raw / raw.sum()
                offset = x + (i - (n_splits - 1) / 2) * bar_width
                bars = ax.bar(offset, heights, bar_width,
                              label=f"{split} (n={len(labels):,}, std={labels.std():.2f})",
                              color=split_colors[split], edgecolor="black", linewidth=0.5)

            ax.set_xticks(x)
            ax.set_xticklabels([str(c) for c in all_classes])
            ax.set_xlabel(task.target_col)
            ax.set_ylabel("Proportion")
        else:
            all_vals = np.concatenate([l.values for l in split_labels.values()])
            bins = np.linspace(all_vals.min(), all_vals.max(), 51)

            for split, labels in split_labels.items():
                ax.hist(labels.values, bins=bins, density=True, alpha=0.5,
                        color=split_colors[split], edgecolor="black", linewidth=0.3,
                        label=f"{split} (n={len(labels):,}, std={labels.std():.2f})")

            ax.set_xlabel(task.target_col)
            ax.set_ylabel("Density")

        ax.legend(fontsize=9)
        fig.tight_layout()
        suffix = "cls" if is_classification else "reg"
        fname = os.path.join(outdir, f"label_dist_{dataset_name}_{task_name}_{suffix}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")

print("Done.")
