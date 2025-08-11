import os
import shutil
import uuid
import json
import queue
import threading
import base64
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rtdetrv2'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))

from rtdetrv2.tools.infer import InitArgs, draw, initModel

'''
    initialize Flask app and SocketIO
'''
app = Flask(__name__)
CORS(app) 

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet') # enable CORS for all origins

app.config['UPLOAD_FOLDER'] = 'inputFile'
app.config['OUTPUT_FOLDER'] = 'outputFile'
app.config['LOG_FOLDER'] = 'log'
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # 16 MB max upload size

# ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['LOG_FOLDER'], exist_ok=True)

frame_queue = queue.Queue(maxsize=10)
result_queue = queue.Queue(maxsize=10)
ui_queue = queue.Queue(maxsize=3)

@app.route('/static_input/<path:filename>')
def serve_input_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

@app.route('/static_output/<path:filename>')
def serve_output_file(filename):
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename))

print(f"DEBUG: Flask App Current Working Directory: {os.getcwd()}")

uploaded_files_info = {} # key: original_filename, value: {'uuid_name': ..., 'file_type': ...}
name_mapping_table = [] # (original_name, uuid_name)
detect_annotations = {} # key: uuid_name, value: list of detected frames/info

# key: session ID (sid), value: True (request interrupt) 或 False
interrupt_requests = {} 

video_format = ['mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI']
image_format = ['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG']

# load language file
def load_language(lang_code):
    try:
        with open(f"lang/{lang_code}.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {} # Fallback to empty dict or default English

# default language: english
lang = load_language("en")

'''
    Supporting functions
'''
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
            f.write(f"{i*fps:2f}sec\n")
    return log_path

def reader_thread(cap, frame_queue):
    while True:
        ret, frame = cap.read()
        if not ret:
            frame_queue.put(None)
            break
        frame_queue.put(frame)

def infer_thread(model, frame_queue, result_queue, orig_size, device):
    while True:
        frame = frame_queue.get()
        if frame is None:
            result_queue.put(None)
            break
        # Preprocess
        frame_resized = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
        im_data = (torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0).unsqueeze(0).to(device)
        # Inference
        output = model(im_data, orig_size)
        result_queue.put((frame, output))

def writer_thread(result_queue, output_video, draw, total_frames, frame_placeholder, preview_size, lang, uuid_name, sid, detect_annotations):
    current_frame = 0
    while True:
        item = result_queue.get()
        if item is None:
            break
        frame, output = item
        labels, boxes, scores = output
        detect_frame, box_count = draw([frame], labels, boxes, scores, 0.35)
        output_video.write(detect_frame)
        # Optionally update UI here, but not every frame
        current_frame += 1
        progress = current_frame / total_frames
        display_frame = cv2.resize(detect_frame, preview_size, interpolation=cv2.INTER_LINEAR)

        _, buffer = cv2.imencode('.jpg', display_frame)
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        if box_count > 0:
            detect_annotations.append(current_frame / total_frames)

        ui_queue.put((encoded_image, progress, current_frame))

# inference logics
def perform_inference(sid, file_path, is_video, output_base_dir, uuid_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # depending on the file type, create the output directory
    if is_video: # vedios
        output_dir_for_file = os.path.join(output_base_dir, uuid_name.split('.')[0])
        os.makedirs(output_dir_for_file, exist_ok=True)
        args = InitArgs(file_path, True, output_dir_for_file, device)
    else: # images
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
        orig_size = torch.tensor([width, height])[None].to(args.device)
        preview_size = (800, height * 800 // width)
        
        # temp path for the output video
        output_video_path = os.path.join(output_dir_for_file, uuid_name)
        fourcc = cv2.VideoWriter_fourcc(*'avc1') 
        output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        current_frame = 0

        reader_th = threading.Thread(target=reader_thread, args=(cap, frame_queue))
        infer_th = threading.Thread(target=infer_thread, args=(model, frame_queue, result_queue, orig_size, args.device))
        writer_th = threading.Thread(target=writer_thread, args=(result_queue, output_video, draw, total_frames, None, preview_size, lang, uuid_name, sid, detect_annotation))

        start_time = time.time()
        reader_th.start()
        infer_th.start()
        writer_th.start()

        while writer_th.is_alive():
            try:
                frame_rgb, progress, current_frame = ui_queue.get(timeout=0.1)
                socketio.emit('inference_progress', {'progress': int(progress*100), 'filename': uuid_name, 'frame':frame_rgb}, room=sid)
                eventlet.sleep(0) 
            except queue.Empty:
                pass
        
        reader_th.join()
        infer_th.join()
        writer_th.join()
        
        cap.release()
        output_video.release()
        return detect_annotation, output_video_path, None, total_frames

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

        return detect_annotation, output_image_path, None, 1 # for images, total_frames is 1

# --- Flask API port ---

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
        else: # images
            input_dir = os.path.join(app.config['UPLOAD_FOLDER'], "photo")
        
        os.makedirs(input_dir, exist_ok=True)
        save_path = os.path.join(input_dir, uuid_name)
        
        try:
            uploaded_file.save(save_path)
            uploaded_files_info[uuid_name] = {
                'original_name': file_name,
                'path': save_path,
                'is_video': is_video,
                'uuid_name': uuid_name # pass uuid to front end for future reference
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
    interrupt_requests[sid] = True # interrupt the inference process for this client

@socketio.on('start_inference_batch')
def handle_start_inference_batch():
    global uploaded_files_info, detect_annotations, name_mapping_table
    
    sid = request.sid # 獲取當前連接的 session ID
    interrupt_requests[sid] = False 

    if not uploaded_files_info:
        socketio.emit('inference_error', {'message': 'No files uploaded for inference.'}, room=sid)
        return

    inference_results = []
    
    files_to_infer = list(uploaded_files_info.items()) 
    
    # start inference event for batch
    socketio.emit('batch_inference_started', {'total_files': len(files_to_infer)}, room=sid)

    for i, (uuid_name, file_info) in enumerate(files_to_infer):
        original_name = file_info['original_name']
        file_path = file_info['path']
        is_video = file_info['is_video']
        
        # strat inference event for each file
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
                "index": i 
            }
            inference_results.append(result_data)
            # for those already inferred, we still need to send progress 100% to the front end
            socketio.emit('inference_progress', {'progress': 100, 'filename': uuid_name}, room=sid)
            socketio.emit('file_inference_completed', result_data, room=sid)
            continue

        start_time = time.time()
        annotations, output_file_absolute_path, error_msg, total_frame = perform_inference(
            sid, file_path, is_video, app.config['OUTPUT_FOLDER'], uuid_name
        )
        end_time = time.time()

        if error_msg == "Inference interrupted":
            result_data = {
                "original_name": original_name,
                "uuid_name": uuid_name,
                "success": False, # interrupted inference is considered a failure
                "message": f"Inference was interrupted for '{original_name}'.",
                "index": i
            }
            inference_results.append(result_data)
            socketio.emit('file_inference_interrupted', result_data, room=sid) 
            break 

        elif error_msg:
            result_data = {
                "original_name": original_name,
                "uuid_name": uuid_name,
                "success": False,
                "message": f"Inference failed for '{original_name}': {error_msg}",
                "index": i 
            }
            inference_results.append(result_data)
            socketio.emit('inference_error', result_data, room=sid) # send error event
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
                "message": f"Inference completed for '{original_name}' in {end_time - start_time:.2f}s. FPS:{total_frame/(end_time - start_time):.2f}",
                "output_path_relative": relative_output_path_for_frontend,
                "log_path": log_path,
                "index": i 
            }
            inference_results.append(result_data)
            socketio.emit('file_inference_completed', result_data, room=sid) # complete event for each file
    
    socketio.emit('batch_inference_completed', {'results': inference_results}, room=sid) # complete event for batch

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
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
        os.makedirs(app.config['LOG_FOLDER'], exist_ok=True)
        
        # delete any existing zip files
        if os.path.exists("output_files.zip"):
            os.remove("output_files.zip")
        if os.path.exists("log_files.zip"):
            os.remove("log_files.zip")

        # reset backend state
        uploaded_files_info = {}
        name_mapping_table = []
        detect_annotations = {}
        
        return jsonify({"success": True, "message": "Cleanup successful."})
    except Exception as e:
        return jsonify({"success": False, "message": f"Cleanup failed: {str(e)}"}), 500

if __name__ == '__main__':
    print(f"DEBUG: Flask App Current Working Directory: {os.getcwd()}")
    socketio.run(app, debug=True, port=5000, allow_unsafe_werkzeug=True) # allow_unsafe_werkzeug only for development purposes