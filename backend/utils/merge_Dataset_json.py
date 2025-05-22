import json
import os

def merge_coco_with_folders(json_path1, folder1, json_path2, folder2, output_path):
    with open(json_path1, 'r') as f1, open(json_path2, 'r') as f2:
        coco1 = json.load(f1)
        coco2 = json.load(f2)

    merged = {
        "images": [],
        "annotations": [],
        "categories": coco1["categories"]  # 假設類別相同
    }

    image_id_offset = max(img["id"] for img in coco1["images"]) + 1
    annotation_id_offset = max(ann["id"] for ann in coco1["annotations"]) + 1

    # 修改路徑
    for img in coco1["images"]:
        img["file_name"] = os.path.join(folder1, img["file_name"])
    for img in coco2["images"]:
        img["id"] += image_id_offset
        img["file_name"] = os.path.join(folder2, img["file_name"])
    for ann in coco2["annotations"]:
        ann["id"] += annotation_id_offset
        ann["image_id"] += image_id_offset

    merged["images"].extend(coco1["images"])
    merged["images"].extend(coco2["images"])
    merged["annotations"].extend(coco1["annotations"])
    merged["annotations"].extend(coco2["annotations"])

    with open(output_path, 'w') as f:
        json.dump(merged, f, indent=4)

    print(f"✅ 合併完成，輸出：{output_path}")

merge_coco_with_folders(
    json_path1=r'D:\Drone_humen_detect\MOBDRONE_split_dataset\annotations\val.json',
    folder1=r'D:\Drone_humen_detect\MOBDRONE_split_dataset\val\images',
    json_path2=r'D:\Drone_humen_detect\WiSARD_VIS-4\annotations\valid_annotations.coco.json',
    folder2=r'D:\Drone_humen_detect\WiSARD_VIS-4\valid',
    output_path=r'D:\Drone_humen_detect\dataset_annotations\valid.json'
)
