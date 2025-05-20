import json

LOG_PATH = r"D:\Drone_humen_detect\output\WiSARD_MOBDRONE\log.txt"
TOP_N = 5

# 每個 metric 的名稱與在 test_coco_eval_bbox 中的 index 對應
METRICS = {
    "AP": 0,
    "AP50": 1,
    "AP75": 2,
    "AP_small": 3,
    "AP_medium": 4,
    "AP_large": 5,
}

def load_log(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if "test_coco_eval_bbox" in line]

def rank_by_metric(logs, metric_name, top_n):
    idx = METRICS[metric_name]
    ranked = []

    for entry in logs:
        epoch = entry.get("epoch")
        coco_eval = entry.get("test_coco_eval_bbox", [])
        if len(coco_eval) > idx:
            ranked.append({
                "epoch": epoch,
                "value": coco_eval[idx],
                "checkpoint": f"checkpoint{epoch:04d}.pth"
            })

    # 依照指標數值降序排序
    ranked.sort(key=lambda x: x["value"], reverse=True)

    print(f"\n🔍 Top {top_n} Epochs sorted by {metric_name}:\n")
    for i, r in enumerate(ranked[:top_n]):
        print(f"{i+1}. Epoch {r['epoch']} | {metric_name}: {r['value']:.4f} | Checkpoint: {r['checkpoint']}")

    best = ranked[0]
    print(f"\n✅ Best checkpoint for {metric_name}: {best['checkpoint']} ({metric_name}: {best['value']:.4f})")

def main():
    logs = load_log(LOG_PATH)

    for metric in METRICS.keys():
        rank_by_metric(logs, metric, TOP_N)

if __name__ == "__main__":
    main()
