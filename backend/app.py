import os
import shutil
import uuid
import json
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import torch
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image
import zipfile
import time
import sys

# 假設 rtdetrv2 和 utils 在 sys.path 中，或者您需要根據實際路徑調整
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rtdetrv2'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))

from rtdetrv2.tools.infer import InitArgs, draw, initModel

app = Flask(__name__)
CORS(app) 

app.config['UPLOAD_FOLDER'] = 'inputFile'
app.config['OUTPUT_FOLDER'] = 'outputFile'
app.config['LOG_FOLDER'] = 'log'
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # 16 MB max upload size

# 確保資料夾存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['LOG_FOLDER'], exist_ok=True)

# 配置 Flask 靜態檔案服務
# 用於提供模型推理後的結果檔案 (outputFile)
app.static_folder = 'outputFile'
app.static_url_path = '/static_output'

# 新增：用於提供已上傳的原始檔案 (inputFile)
# 這裡我們使用一個不同的端點名稱，例如 '/static_input'
# 這個路由需要手動註冊，因為 Flask 的 `static_folder` 只能設定一個
@app.route('/static_input/<path:filename>')
def serve_input_file(filename):
    # 這裡的 filename 包含了子目錄（例如 photo/image.jpg 或 uuid_folder/video.mp4）
    # 我們需要從 inputFile 中找到它
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))


# 模擬 session_state，在真實生產環境應使用更持久的儲存（如資料庫）
# 或者每個請求獨立處理，避免共享狀態
uploaded_files_info = {} # key: original_filename, value: {'uuid_name': ..., 'file_type': ...}
name_mapping_table = [] # (original_name, uuid_name)
detect_annotations = {} # key: uuid_name, value: list of detected frames/info

video_format = ['mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI']
image_format = ['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG']

# 載入語言檔 (可選擇性保留，或將訊息直接硬編碼在前端)
def load_language(lang_code):
    try:
        with open(f"lang/{lang_code}.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {} # Fallback to empty dict or default English

# 假設語言預設為英文
lang = load_language("en")

# --- 輔助函數 (從 main.py 提取並修改) ---

def find_uuid_name(name, mapping_table):
    for old_name, new_name in mapping_table:
        if name == old_name:
            return new_name
    return None

def change_name_to_uuid(file_name, mapping_table):
    finding_result = find_uuid_name(file_name, mapping_table)
    if finding_result is None:
        new_name = str(uuid.uuid4()) + '.' + file_name.split('.')[-1]
        mapping_table.append((file_name, new_name))
        return new_name, mapping_table
    else:
        return finding_result, mapping_table

def recover_name(uuid_name, mapping_table, file_type):
    for old_name, new_name in mapping_table:
        if uuid_name == new_name:
            # 這是獲取原始檔案名的邏輯
            original_file_name = old_name 
            
            # 構建恢復後的檔案路徑
            if file_type == "video":
                old_base_name = old_name.split('.')[0]
                new_base_name = new_name.split('.')[0]
                
                # 重命名 output 資料夾和檔案
                old_folder_path = os.path.join(app.config['OUTPUT_FOLDER'], new_base_name)
                new_folder_path = os.path.join(app.config['OUTPUT_FOLDER'], old_base_name)
                
                if os.path.exists(old_folder_path):
                    shutil.move(old_folder_path, new_folder_path)
                    
                old_file_path = os.path.join(new_folder_path, new_name)
                new_file_path = os.path.join(new_folder_path, old_name)
                
                if os.path.exists(old_file_path):
                    os.rename(old_file_path, new_file_path)
                
                return original_file_name, new_file_path
            else: # photo
                old_file_path = os.path.join(app.config['OUTPUT_FOLDER'], 'photo', new_name)
                new_file_path = os.path.join(app.config['OUTPUT_FOLDER'], 'photo', old_name)
                if os.path.exists(old_file_path):
                    os.rename(old_file_path, new_file_path)
                return original_file_name, new_file_path
    return None, None # Should not happen if uuid_name is valid


def make_log(annotations, fps, file_name_base):
    os.makedirs(app.config['LOG_FOLDER'], exist_ok=True)
    log_path = os.path.join(app.config['LOG_FOLDER'], file_name_base + '.txt')
    with open(log_path, 'w') as f:
        f.write(f"Detected people frames for {file_name_base}:\n")
        for i in annotations:
            f.write(f"{i/fps:.2f}sec\n")
    return log_path

# --- 模型推斷邏輯 (從 main.py 的 infer 函數提取並修改) ---
def perform_inference(file_path, is_video, output_base_dir, uuid_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 根據是否為影片，調整 output_path
    if is_video:
        output_dir_for_file = os.path.join(output_base_dir, uuid_name.split('.')[0])
        os.makedirs(output_dir_for_file, exist_ok=True)
        args = InitArgs(file_path, True, output_dir_for_file, device)
    else: # 圖片
        output_dir_for_file = os.path.join(output_base_dir, "photo")
        os.makedirs(output_dir_for_file, exist_ok=True)
        args = InitArgs(file_path, False, output_dir_for_file, device)
        
    model = initModel(args)
    
    detect_annotation = []
    
    if is_video:
        cap = cv2.VideoCapture(file_path) 
        if not cap.isOpened():
            print(f"Error: Could not open video {file_path}")
            return [], None, "Error: Could not open video."

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 輸出影片的暫存路徑
        output_video_path = os.path.join(output_dir_for_file, uuid_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 使用 mp4v 編碼器
        output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        current_frame = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_resized = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
            im_data = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
            im_data = im_data.unsqueeze(0).to(device)
            orig_size = torch.tensor([frame.shape[1], frame.shape[0]])[None].to(device)

            output = model(im_data, orig_size)
            labels, boxes, scores = output
            
            im_pil = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
            detect_frame, box_count = draw([im_pil], labels, boxes, scores, 0.35)
            frame_out = cv2.cvtColor(np.array(detect_frame), cv2.COLOR_RGB2BGR)

            frame_out_resized = cv2.resize(frame_out, (width, height), interpolation=cv2.INTER_LINEAR)
            output_video.write(frame_out_resized)
            
            if box_count > 0:
                detect_annotation.append(current_frame)
            current_frame += 1
        
        cap.release()
        output_video.release()
        return detect_annotation, output_video_path, None

    else: # Image
        img = cv2.imread(file_path)
        if img is None:
            return [], None, "Error: Could not read image."
            
        im_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(device)
        
        transforms = T.Compose([
            T.Resize((640, 640)),  
            T.ToTensor(),
        ])
        im_data = transforms(im_pil)[None].to(device)
        output = model(im_data, orig_size)
        labels, boxes, scores = output
        
        detect_frame, box_count = draw([im_pil], labels, boxes, scores, 0.35)
        frame_out = cv2.cvtColor(np.array(detect_frame), cv2.COLOR_RGB2BGR)
        
        output_image_path = os.path.join(output_dir_for_file, uuid_name)
        cv2.imwrite(output_image_path, frame_out)
        return detect_annotation, output_image_path, None

# --- Flask API 接口 ---

@app.route('/')
def home():
    return "Backend is running. Please open frontend/index.html in your browser."

@app.route('/upload', methods=['POST'])
def upload_file():
    global uploaded_files_info, name_mapping_table

    if 'files[]' not in request.files:
        return jsonify({"success": False, "message": "No file part in the request"}), 400

    files = request.files.getlist('files[]')
    
    uploaded_results = []
    
    for uploaded_file in files:
        if uploaded_file.filename == '':
            continue
        
        file_name = secure_filename(uploaded_file.filename)
        file_extension = file_name.split('.')[-1]
        
        is_video = file_extension in video_format
        is_image = file_extension in image_format

        if not (is_video or is_image):
            uploaded_results.append({
                "original_name": file_name,
                "success": False,
                "message": f"Unsupported format: {file_name}"
            })
            continue

        uuid_name, name_mapping_table = change_name_to_uuid(file_name, name_mapping_table)
        
        if is_video:
            input_dir = os.path.join(app.config['UPLOAD_FOLDER'], uuid_name.split('.')[0])
        else: # 圖片
            input_dir = os.path.join(app.config['UPLOAD_FOLDER'], "photo")
        
        os.makedirs(input_dir, exist_ok=True)
        save_path = os.path.join(input_dir, uuid_name)
        
        try:
            uploaded_file.save(save_path)
            uploaded_files_info[uuid_name] = {
                'original_name': file_name,
                'path': save_path,
                'is_video': is_video,
                'uuid_name': uuid_name # 將 uuid_name 也傳給前端，方便構建 URL
            }
            uploaded_results.append({
                "original_name": file_name,
                "uuid_name": uuid_name,
                "is_video": is_video,
                "success": True,
                "message": f"File '{file_name}' uploaded successfully."
            })
        except Exception as e:
            uploaded_results.append({
                "original_name": file_name,
                "success": False,
                "message": f"Failed to save file '{file_name}': {str(e)}"
            })

    return jsonify({"success": True, "results": uploaded_results})

@app.route('/infer', methods=['POST'])
def start_inference():
    global uploaded_files_info, detect_annotations, name_mapping_table

    if not uploaded_files_info:
        return jsonify({"success": False, "message": "No files uploaded for inference."}), 400

    inference_results = []
    
    for uuid_name, file_info in uploaded_files_info.items():
        original_name = file_info['original_name']
        file_path = file_info['path']
        is_video = file_info['is_video']
        
        if uuid_name in detect_annotations and detect_annotations[uuid_name] is not None:
            # 避免重複推斷
            # 重新獲取恢復後的路徑，以確保即使重新整理也能正確顯示
            original_name_recovered, final_output_path = recover_name(
                uuid_name, name_mapping_table, "video" if is_video else "photo"
            )
            
            # 如果原始日誌檔名存在，則重建日誌檔名
            log_path = None
            if original_name_recovered:
                log_path = os.path.join(app.config['LOG_FOLDER'], original_name_recovered.split('.')[0] + '.txt')

            inference_results.append({
                "original_name": original_name,
                "uuid_name": uuid_name,
                "success": True,
                "message": f"File '{original_name}' already inferred.",
                "output_path": final_output_path, 
                "log_path": log_path,
                "is_video": is_video
            })
            continue

        start_time = time.time()
        annotations, output_file_path, error_msg = perform_inference(
            file_path, is_video, app.config['OUTPUT_FOLDER'], uuid_name
        )
        end_time = time.time()
        
        if error_msg:
            inference_results.append({
                "original_name": original_name,
                "uuid_name": uuid_name,
                "success": False,
                "message": f"Inference failed for '{original_name}': {error_msg}"
            })
        else:
            detect_annotations[uuid_name] = annotations
            
            # 恢復原始檔名並獲取最終輸出路徑
            original_name_recovered, final_output_path = recover_name(
                uuid_name, name_mapping_table, "video" if is_video else "photo"
            )
            
            log_path = None
            if is_video and original_name_recovered:
                # 獲取影片的 fps 用於 log
                cap = cv2.VideoCapture(file_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                log_path = make_log(annotations, fps, original_name_recovered.split('.')[0])
            
            inference_results.append({
                "original_name": original_name,
                "uuid_name": uuid_name,
                "success": True,
                "message": f"Inference completed for '{original_name}' in {end_time - start_time:.2f}s.",
                "output_path": final_output_path, # 提供前端展示和下載
                "log_path": log_path,
                "is_video": is_video
            })
    
    return jsonify({"success": True, "results": inference_results})

@app.route('/download_output_zip')
def download_output_zip():
    zip_path = "output_files.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, dirs, files in os.walk(app.config['OUTPUT_FOLDER']):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, app.config['OUTPUT_FOLDER'])
                zipf.write(full_path, arcname)
    return send_file(zip_path, as_attachment=True, download_name='output_files.zip')

@app.route('/download_log_zip')
def download_log_zip():
    log_zip_path = "log_files.zip"
    with zipfile.ZipFile(log_zip_path, 'w') as zipf:
        for root, dirs, files in os.walk(app.config['LOG_FOLDER']):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, app.config['LOG_FOLDER'])
                zipf.write(full_path, arcname)
    return send_file(log_zip_path, as_attachment=True, download_name='log_files.zip')

@app.route('/cleanup', methods=['POST'])
def cleanup_files():
    global uploaded_files_info, name_mapping_table, detect_annotations
    try:
        shutil.rmtree(app.config['UPLOAD_FOLDER'], ignore_errors=True)
        shutil.rmtree(app.config['OUTPUT_FOLDER'], ignore_errors=True)
        shutil.rmtree(app.config['LOG_FOLDER'], ignore_errors=True)
        # 重新創建空資料夾
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
        os.makedirs(app.config['LOG_FOLDER'], exist_ok=True)
        
        # 刪除壓縮檔
        if os.path.exists("output_files.zip"):
            os.remove("output_files.zip")
        if os.path.exists("log_files.zip"):
            os.remove("log_files.zip")

        # 重置後端狀態
        uploaded_files_info = {}
        name_mapping_table = []
        detect_annotations = {}
        
        return jsonify({"success": True, "message": "Cleanup successful."})
    except Exception as e:
        return jsonify({"success": False, "message": f"Cleanup failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)