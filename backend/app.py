import os
import shutil
import uuid
import json
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import eventlet
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

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet') # 設置 CORS 允許所有來源

app.config['UPLOAD_FOLDER'] = 'inputFile'
app.config['OUTPUT_FOLDER'] = 'outputFile'
app.config['LOG_FOLDER'] = 'log'
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # 16 MB max upload size

# 確保資料夾存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['LOG_FOLDER'], exist_ok=True)

# 新增：用於提供已上傳的原始檔案 (inputFile)
@app.route('/static_input/<path:filename>')
def serve_input_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

@app.route('/static_output/<path:filename>')
def serve_output_file(filename):
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename))

print(f"DEBUG: Flask App Current Working Directory: {os.getcwd()}")
# print(f"DEBUG: Flask static_folder path: {os.path.abspath(app.static_folder)}")

# 模擬 session_state，在真實生產環境應使用更持久的儲存（如資料庫）
uploaded_files_info = {} # key: original_filename, value: {'uuid_name': ..., 'file_type': ...}
name_mapping_table = [] # (original_name, uuid_name)
detect_annotations = {} # key: uuid_name, value: list of detected frames/info

# key: session ID (sid), value: True (表示請求中斷) 或 False
interrupt_requests = {} 

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
            original_file_name = old_name

            if file_type == "video":
                old_base_name = old_name.split('.')[0]
                new_base_name = new_name.split('.')[0]

                old_output_folder_path = os.path.join(app.config['OUTPUT_FOLDER'], new_base_name)
                new_output_folder_path = os.path.join(app.config['OUTPUT_FOLDER'], old_base_name)

                if os.path.exists(old_output_folder_path):
                    shutil.move(old_output_folder_path, new_output_folder_path)

                old_output_file_path = os.path.join(new_output_folder_path, new_name)
                new_output_file_path = os.path.join(new_output_folder_path, old_name)

                if os.path.exists(old_output_file_path):
                    os.rename(old_output_file_path, new_output_file_path)

                relative_path_for_frontend = os.path.join(old_base_name, old_name).replace('\\', '/')
                return original_file_name, relative_path_for_frontend

            else: # photo
                old_output_file_path = os.path.join(app.config['OUTPUT_FOLDER'], 'photo', new_name)
                new_output_file_path = os.path.join(app.config['OUTPUT_FOLDER'], 'photo', old_name)

                if os.path.exists(old_output_file_path):
                    os.rename(old_output_file_path, new_output_file_path)

                relative_path_for_frontend = os.path.join('photo', old_name).replace('\\', '/')
                return original_file_name, relative_path_for_frontend
    return None, None


def make_log(annotations, fps, file_name_base):
    os.makedirs(app.config['LOG_FOLDER'], exist_ok=True)
    log_path = os.path.join(app.config['LOG_FOLDER'], file_name_base + '.txt')
    with open(log_path, 'w') as f:
        f.write(f"Detected people frames for {file_name_base}:\n")
        for i in annotations:
            f.write(f"{i/fps:.2f}sec\n")
    return log_path

# --- 模型推斷邏輯 (從 main.py 的 infer 函數提取並修改) ---
def perform_inference(sid, file_path, is_video, output_base_dir, uuid_name):
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
            socketio.emit('inference_error', {'message': f'Failed to open video: {file_path}'}, room=sid)
            return [], None, "Error: Could not open video."

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 輸出影片的暫存路徑
        output_video_path = os.path.join(output_dir_for_file, uuid_name)
        fourcc = cv2.VideoWriter_fourcc(*'avc1') 
        output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        current_frame = 0
        while cap.isOpened():
                if interrupt_requests.get(sid): # 檢查當前 sid 是否請求中斷
                    print(f"Inference interrupted for {uuid_name} by client {sid}")
                    socketio.emit('inference_interrupted', {'message': f'Inference interrupted for {uuid_name}.', 'uuid_name': uuid_name}, room=sid)
                    cap.release()
                    output_video.release()
                    # 在中斷時清除該 sid 的中斷標誌
                    interrupt_requests[sid] = False 
                    return [], None, "Inference interrupted" # 返回中斷狀態

                t0 = time.perf_counter()
                ret, frame = cap.read()
                t1 = time.perf_counter()
                if not ret:
                    break

                # Preprocessing
                t2 = time.perf_counter()
                frame_resized = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
                im_data = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
                im_data = im_data.unsqueeze(0).to(args.device)
                orig_size = torch.tensor([frame.shape[1], frame.shape[0]])[None].to(args.device)
                t3 = time.perf_counter()

                # Model inference
                output = model(im_data, orig_size)
                t4 = time.perf_counter()

                # Postprocessing/drawing
                labels, boxes, scores = output
                detect_frame, box_count = draw([frame], labels, boxes, scores, 0.35)
                frame_out = cv2.cvtColor(np.array(detect_frame), cv2.COLOR_RGB2BGR)
                t5 = time.perf_counter()

                #frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(cv2.resize(frame_out, (800, 600), interpolation=cv2.INTER_LINEAR))
                t6 = time.perf_counter()

                # Output video write
                output_video.write(detect_frame)
                t7 = time.perf_counter()

                # Print timings
                print(f"Read: {t1-t0:.3f}s, Pre: {t3-t2:.3f}s, Infer: {t4-t3:.3f}s, Draw: {t5-t4:.3f}s, UI: {t6-t5:.3f}s, Write: {t7-t6:.3f}s")

                # Progress bar
                current_frame += 1
                progress_percent = int((current_frame / total_frames) * 100)
                socketio.emit('inference_progress', {'progress': progress_percent, 'filename': uuid_name}, room=sid)
                eventlet.sleep(0)

                # Collect the frame that is detected
                if  box_count > 0:
                    detect_annotation.append(current_frame)
        
        cap.release()
        output_video.release()
        return detect_annotation, output_video_path, None

    else: # Image
        img = cv2.imread(file_path)
        if img is None:
            socketio.emit('inference_error', {'message': f'Failed to read image: {file_path}'}, room=sid)
            return [], None, "Error: Could not read image."
        
        if interrupt_requests.get(sid):
            print(f"Inference interrupted for {uuid_name} by client {sid}")
            socketio.emit('inference_interrupted', {'message': f'Inference interrupted for {uuid_name}.', 'uuid_name': uuid_name}, room=sid)
            interrupt_requests[sid] = False
            return [], None, "Inference interrupted"
            
        start_time = time.time()
        image_resized = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
        im_data = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        im_data = im_data.unsqueeze(0).to(args.device)
        orig_size = torch.tensor([img.shape[1], img.shape[0]])[None].to(args.device)

        output = model(im_data, orig_size)
        labels, boxes, scores = output
        detect_frame, box_count = draw([img], labels, boxes, scores, 0.35)

        output_image_path = os.path.join(output_dir_for_file, uuid_name)
        cv2.imwrite(output_image_path, detect_frame)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Image inference time: {elapsed_time:.2f} seconds")

        socketio.emit('inference_progress', {'progress': 100, 'filename': uuid_name}, room=sid)
        eventlet.sleep(0)

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

@socketio.on('connect')
def test_connect():
    print('Client connected:', request.sid)
    emit('my response', {'data': 'Connected'})

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected:', request.sid)

@socketio.on('interrupt_inference')
def handle_interrupt_inference():
    sid = request.sid
    print(f"Client {sid} requested interruption.")
    interrupt_requests[sid] = True # 設置中斷標誌

@socketio.on('start_inference_batch') # 接收前端發送的批次推理請求
def handle_start_inference_batch():
    global uploaded_files_info, detect_annotations, name_mapping_table
    
    sid = request.sid # 獲取當前連接的 session ID
    interrupt_requests[sid] = False 

    if not uploaded_files_info:
        socketio.emit('inference_error', {'message': 'No files uploaded for inference.'}, room=sid)
        return

    inference_results = []
    
    # 這裡使用 list(uploaded_files_info.items()) 複製一份，防止在迭代時被修改
    files_to_infer = list(uploaded_files_info.items()) 
    
    # 發送批次推理開始事件
    socketio.emit('batch_inference_started', {'total_files': len(files_to_infer)}, room=sid)

    for i, (uuid_name, file_info) in enumerate(files_to_infer):
        original_name = file_info['original_name']
        file_path = file_info['path']
        is_video = file_info['is_video']
        
        # 發送當前檔案的推理開始事件
        socketio.emit('file_inference_started', {'filename': original_name, 'index': i}, room=sid)

        if uuid_name in detect_annotations and detect_annotations[uuid_name] is not None:
            original_name_recovered, relative_output_path_for_frontend = recover_name(
                uuid_name, name_mapping_table, "video" if is_video else "photo"
            )

            log_path = None
            if original_name_recovered and is_video:
                log_path = os.path.join(app.config['LOG_FOLDER'], original_name_recovered.split('.')[0] + '.txt')

            result_data = {
                "original_name": original_name,
                "uuid_name": uuid_name,
                "is_video": is_video,
                "success": True,
                "message": f"File '{original_name}' already inferred.",
                "output_path_relative": relative_output_path_for_frontend,
                "log_path": log_path,
                "index": i # 添加索引方便前端對應
            }
            inference_results.append(result_data)
            # 對於已推斷過的檔案，直接發送進度 100% 和完成事件
            socketio.emit('inference_progress', {'progress': 100, 'filename': uuid_name}, room=sid)
            socketio.emit('file_inference_completed', result_data, room=sid)
            continue

        start_time = time.time()
        # perform_inference 現在需要 sid 來發送進度更新
        annotations, output_file_absolute_path, error_msg = perform_inference(
            sid, file_path, is_video, app.config['OUTPUT_FOLDER'], uuid_name
        )
        end_time = time.time()

        if error_msg == "Inference interrupted":
            result_data = {
                "original_name": original_name,
                "uuid_name": uuid_name,
                "success": False, # 中斷也算不成功完成
                "message": f"Inference was interrupted for '{original_name}'.",
                "index": i
            }
            inference_results.append(result_data)
            # 發送給前端，讓前端知道這個文件被中斷了
            socketio.emit('file_inference_interrupted', result_data, room=sid) 
            # 如果是中斷，則跳出整個批次推理迴圈
            break 
        elif error_msg:
            result_data = {
                "original_name": original_name,
                "uuid_name": uuid_name,
                "success": False,
                "message": f"Inference failed for '{original_name}': {error_msg}",
                "index": i # 添加索引方便前端對應
            }
            inference_results.append(result_data)
            socketio.emit('inference_error', result_data, room=sid) # 發送錯誤事件
        else:
            detect_annotations[uuid_name] = annotations

            original_name_recovered, relative_output_path_for_frontend = recover_name(
                uuid_name, name_mapping_table, "video" if is_video else "photo"
            )

            log_path = None
            if is_video and original_name_recovered:
                cap = cv2.VideoCapture(file_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                log_path = make_log(annotations, fps, original_name_recovered.split('.')[0])

            result_data = {
                "original_name": original_name,
                "uuid_name": uuid_name,
                "is_video": is_video,
                "success": True,
                "message": f"Inference completed for '{original_name}' in {end_time - start_time:.2f}s.",
                "output_path_relative": relative_output_path_for_frontend,
                "log_path": log_path,
                "index": i # 添加索引方便前端對應
            }
            inference_results.append(result_data)
            socketio.emit('file_inference_completed', result_data, room=sid) # 發送單個檔案完成事件
    
    socketio.emit('batch_inference_completed', {'results': inference_results}, room=sid) # 發送批次推理完成事件

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

        # 這裡的邏輯需要調整，確保即使是已推斷過的，也能正確地返回相對路徑
        # 如果 detect_annotations[uuid_name] 不為 None，表示已經推斷過
        # 我們直接從 recover_name 獲取其最終的相對路徑
        if uuid_name in detect_annotations and detect_annotations[uuid_name] is not None:
            # 避免重複推斷，但仍返回正確的輸出路徑
            original_name_recovered, relative_output_path_for_frontend = recover_name(
                uuid_name, name_mapping_table, "video" if is_video else "photo"
            )

            log_path = None
            if original_name_recovered and is_video:
                log_path = os.path.join(app.config['LOG_FOLDER'], original_name_recovered.split('.')[0] + '.txt')

            inference_results.append({
                "original_name": original_name,
                "uuid_name": uuid_name,
                "is_video": is_video, # 新增這個，讓前端知道是圖片還是影片
                "success": True,
                "message": f"File '{original_name}' already inferred.",
                "output_path_relative": relative_output_path_for_frontend, # 返回相對路徑
                "log_path": log_path,
            })
            continue

        start_time = time.time()
        annotations, output_file_absolute_path, error_msg = perform_inference(
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

            # recover_name 會處理檔名恢復和資料夾重命名，並返回用於前端的相對路徑
            original_name_recovered, relative_output_path_for_frontend = recover_name(
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
                "is_video": is_video, # 新增這個，讓前端知道是圖片還是影片
                "success": True,
                "message": f"Inference completed for '{original_name}' in {end_time - start_time:.2f}s.",
                "output_path_relative": relative_output_path_for_frontend, # 返回相對路徑
                "log_path": log_path,
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
    print(f"DEBUG: Flask App Current Working Directory: {os.getcwd()}")
    socketio.run(app, debug=True, port=5000, allow_unsafe_werkzeug=True) # allow_unsafe_werkzeug 僅用於開發環境