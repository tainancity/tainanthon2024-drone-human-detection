import json
import os
import matplotlib.pyplot as plt

# 設定 log 路徑
LOG_PATH = r"D:\Drone_humen_detect\output\WiSARD_MOBDRONE\log.txt"
SAVE_DIR = os.path.dirname(LOG_PATH)

def load_metrics(log_path):
    epochs = []
    ap, ap50, ap75 = [], [], []
    ap_small, ap_medium, ap_large = [], [], []
    recall = []

    with open(log_path, "r") as f:
        for line in f:
            data = json.loads(line)
            if "test_coco_eval_bbox" in data:
                metrics = data["test_coco_eval_bbox"]
                if len(metrics) >= 12:
                    epochs.append(data["epoch"])
                    ap.append(metrics[0])
                    ap50.append(metrics[1])
                    ap75.append(metrics[2])
                    ap_small.append(metrics[3])
                    ap_medium.append(metrics[4])
                    ap_large.append(metrics[5])
                    recall.append(metrics[12] if len(metrics) > 12 else None)  # 若有 Recall（可能不存在）

    return epochs, ap, ap50, ap75, ap_small, ap_medium, ap_large, recall

def plot_and_save(x, y_list, labels, title, ylabel, filename):
    plt.figure(figsize=(10, 6))
    for y, label in zip(y_list, labels):
        plt.plot(x, y, label=label, marker='o')

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, filename))
    plt.close()

def main():
    epochs, ap, ap50, ap75, ap_s, ap_m, ap_l, recall = load_metrics(LOG_PATH)

    # 1️⃣ mAP, AP50, AP75
    plot_and_save(
        epochs,
        [ap, ap50, ap75],
        ["mAP@[.5:.95]", "AP50", "AP75"],
        "AP Metrics Curve",
        "AP",
        "map_ap50_ap75.png"
    )

    # 2️⃣ AP_small, AP_medium, AP_large
    plot_and_save(
        epochs,
        [ap_s, ap_m, ap_l],
        ["AP_small", "AP_medium", "AP_large"],
        "Object Size Specific AP Curve",
        "AP",
        "ap_small_medium_large.png"
    )

    # 3️⃣ Recall (if available)
    if any(r is not None for r in recall):
        plot_and_save(
            epochs,
            [recall],
            ["Recall"],
            "Recall Curve",
            "Recall",
            "recall_curve.png"
        )
        print("✅ Saved: recall_curve.png")
    else:
        print("⚠️ Recall data not found in log.txt.")

    print("✅ Saved: map_ap50_ap75.png, ap_small_medium_large.png")

if __name__ == "__main__":
    main()
