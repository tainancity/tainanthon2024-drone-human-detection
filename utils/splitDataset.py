# import os
# import json
# import random
# import shutil

# # === 你的參數 ===
# coco_path = r'D:\Drone_humen_detect\MOBDrone_videos\video_split_fullhd\annotations_person_coco_classes.json'
# images_dir = r'D:\Drone_humen_detect\MOBDrone_videos\video_split_fullhd\frames'
# output_dir = r'MOBDRONE_split_dataset'
# train_ratio = 0.8
# seed = 42

# # === 載入 COCO JSON ===
# with open(coco_path, 'r') as f:
#     coco = json.load(f)

# images = coco['images']
# annotations = coco['annotations']
# categories = coco['categories']

# # === 清理掉找不到的圖片 ===
# valid_images = []
# valid_image_ids = set()
# missing_files = []

# for img in images:
#     img_path = os.path.join(images_dir, img['file_name'])
#     if os.path.exists(img_path):
#         valid_images.append(img)
#         valid_image_ids.add(img['id'])
#     else:
#         missing_files.append(img['file_name'])

# valid_annotations = [ann for ann in annotations if ann['image_id'] in valid_image_ids]

# print(f"✅ 有效圖片數量: {len(valid_images)}")
# print(f"❌ 缺失圖片數量: {len(missing_files)}")
# if missing_files:
#     print("➡️ 缺失範例:", missing_files[:5])

# # === 切 train / val ===
# random.seed(seed)
# random.shuffle(valid_images)
# total = len(valid_images)
# train_count = int(train_ratio * total)

# train_images = valid_images[:train_count]
# val_images = valid_images[train_count:]

# # === 儲存各 split ===
# def save_split(split_name, split_images):
#     # 圖片輸出路徑
#     split_img_dir = os.path.join(output_dir, split_name, 'images')
#     os.makedirs(split_img_dir, exist_ok=True)

#     # annotations 過濾
#     image_ids = set(img['id'] for img in split_images)
#     split_annotations = [ann for ann in valid_annotations if ann['image_id'] in image_ids]

#     # 複製圖像
#     for img in split_images:
#         src = os.path.join(images_dir, img['file_name'])
#         dst = os.path.join(split_img_dir, img['file_name'])
#         if os.path.exists(src):
#             shutil.copyfile(src, dst)

#     # 存 annotations 到共用 annotations 資料夾
#     annotation_dir = os.path.join(output_dir, 'annotations')
#     os.makedirs(annotation_dir, exist_ok=True)

#     split_json = {
#         'images': split_images,
#         'annotations': split_annotations,
#         'categories': categories
#     }

#     out_json_path = os.path.join(annotation_dir, f'{split_name}.json')
#     with open(out_json_path, 'w') as f:
#         json.dump(split_json, f)

#     print(f"✅ {split_name}: {len(split_images)} 張圖, {len(split_annotations)} 筆標註 -> {out_json_path}")

# # === 執行 ===
# save_split('train', train_images)
# save_split('val', val_images)




from roboflow import Roboflow
rf = Roboflow(api_key="5knojipjkgDebxYFV8jr")
project = rf.workspace("custom-dataset-for-ft").project("custom-ft-dataset")
version = project.version(3)
dataset = version.download("coco")
                