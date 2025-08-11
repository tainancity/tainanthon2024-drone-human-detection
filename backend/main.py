import streamlit as st
import os
import torch
import torchvision.transforms as T
import numpy as np 
from PIL import Image
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import cv2
import zipfile
import shutil
import uuid
from rtdetrv2.tools.infer import InitArgs, draw, initModel
from utils.stream import stream
import time
import locale
import ctypes
import ffmpegcv
import threading
import queue

video_format = ['mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI']
image_format = ['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG']

import json

def load_language(lang):
    try:
        with open(f"lang/{lang}.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    
def main():

    st.title(lang.get("title"))
    
    # upload the file
    uploaded_files = st.file_uploader(lang.get("upload_button"), type=['mp4', 'mov', 'avi', 'png', 'jpg', 'jpeg'], accept_multiple_files=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Video_Type = []
    if uploaded_files is not None:
        # reset session state variables
        if 'last_uploaded_files' not in st.session_state:
            st.session_state.last_uploaded_files = []
        if 'detect_annotations' not in st.session_state:
            st.session_state.detect_annotations = {}
        if 'infer_correct' not in st.session_state:
            st.session_state.infer_correct = False
        if 'has_infer_result' not in st.session_state:
            st.session_state.has_infer_result = False
        if 'name_mapping_table' not in st.session_state:
            st.session_state.name_mapping_table = []
        if 'completed_files' not in st.session_state:
            st.session_state.completed_files = []
        name_mapping_table = st.session_state.name_mapping_table
        completed_files = st.session_state.completed_files

        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            file_extension = file_name.split('.')[-1]
            
            # classify the input file is video or image
            if file_extension in video_format:
                Video_Type.append(True)     # inputs video type
            elif file_extension in image_format:
                Video_Type.append(False)
            else:
                st.warning(lang.get("unsupported_format").format(file_name=file_name))
                continue

            if file_name not in st.session_state.last_uploaded_files:
                st.session_state.detect_annotations[file_name] = None
                st.session_state.last_uploaded_files.append(file_name)

            upload_success = st.success(lang.get("file_uploaded").format(file_name=file_name))
            if file_extension in image_format:
                st.image(uploaded_file)
            elif file_extension in video_format:
                st.video(uploaded_file)
            else:
                st.warning(lang.get("unsupported_format").format(file_name=file_name))
                continue

            # create dir of to save the input file and inference outcome
            base_name = file_name.split('.')[0]
            if os.path.exists("outputfile") and base_name in os.listdir("outputFile") and st.session_state.has_infer_result is False:
                st.warning(lang.get("file_name_conflict").format(file_name=file_name))
                continue
            uuid_name, name_mapping_table = change_name_to_uuid(file_name, name_mapping_table)

            if Video_Type[-1]:  # videos
                input_dir = os.path.join("inputFile", uuid_name.split('.')[0])
            else:  # images
                input_dir = os.path.join("inputFile", "photo")
            output_dir = os.path.join("outputFile", uuid_name.split('.')[0])
            
            try:
                os.makedirs(input_dir, exist_ok=True)
                if Video_Type[-1]:  # images need output directory
                    os.makedirs(output_dir, exist_ok=True)
            except Exception as e:
                st.error(lang.get("dir_creation_failed").format(e=e))
                continue

            # copy the video to inputFile
            save_path = os.path.join(input_dir, uuid_name)
            try:
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                fps = cv2.VideoCapture(save_path).get(cv2.CAP_PROP_FPS)
            except Exception as e:
                st.error(lang.get("file_save_failed").format(e=e))
                
            # close the success message    
            upload_success.empty()
            
            save_success = st.success(lang.get("file_saved").format(save_path=save_path))
            save_success.empty()
        
        # show staert inference button
        if st.session_state.infer_correct == False and st.session_state.last_uploaded_files != []:
            st.session_state.infer_correct = st.button(lang.get("infer_button"))

        # Start to inference if not already done
        if st.session_state.infer_correct:
            for i, uploaded_file in enumerate(uploaded_files):
                file_name = uploaded_file.name
                if st.session_state.detect_annotations[file_name] is None:
                    uuid_name = find_uuid_name(file_name, name_mapping_table)
                    base_name = uuid_name.split('.')[0]
                    file_extension = file_name.split('.')[-1]
                    file_type = "video" if Video_Type[i] else "photo"
                    save_path = f"inputFile/{base_name}/{uuid_name}"
                    output_path = f"outputFile/{uuid_name.split('.')[0]}"
                    if not Video_Type[i]:   
                        save_path = f"inputFile/photo/{uuid_name}"
                        output_path = f"outputFile/photo"
                    args = InitArgs(save_path, Video_Type[i], output_path, device)
                    model = initModel(args)
                    st.session_state.detect_annotations[file_name] = infer(args, model, base_name, file_extension)
                    original_name, new_output_path = recover_name(uuid_name, name_mapping_table, file_type)
                    if file_extension in video_format:
                        log_path = make_log(st.session_state.detect_annotations[file_name], fps, original_name.split('.')[0])
                        st.session_state.completed_files.append(new_output_path)
                        st.success(lang.get("file_saved").format(save_path=log_path))
            st.session_state.infer_correct = False
            st.session_state.has_infer_result = True

        if st.session_state.has_infer_result:
            for i, file_name in enumerate(completed_files):
                st.video(file_name)

        if st.session_state.has_infer_result:
            # download zipped output files
            zip_path = zip_output_files()
            with open(zip_path, "rb") as f:
                st.download_button(lang.get("download_zip"), f, 'output_files.zip')
            
            # download zipped log files
            log_zip_path = zip_log_files()
            with open(log_zip_path, "rb") as f:
                st.download_button(lang.get("download_log"), f, 'log_files.zip')

            # cleanup temp files 
    st.button(lang.get("cleanup_button"), on_click=lambda: cleanup_files())

#
#   Zip output files function:
#       parameters: None
#       function:
#           1. Zip all the output files in the outputFile directory
#           2. Return the path of the zip file
#
def zip_output_files():
    zip_path = "output_files.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, dirs, files in os.walk("outputFile"):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), "outputFile"))
    return zip_path

#
#   Zip log files function:
#       parameters: None
#       function:
#           1. Zip all the log files in the log directory
#           2. Return the path of the zip file
#
def zip_log_files():
    log_zip_path = "log_files.zip"
    with zipfile.ZipFile(log_zip_path, 'w') as zipf:
        for root, dirs, files in os.walk("log"):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), "log"))
    return log_zip_path

#
#   Cleanup_files function:
#       parameters: None
#       function:
#           1. Remove the inputFile, outputFile, and log directories
#           2. Remove the output_files.zip and log_files.zip
#           3. Reset the session state variables
#           4. Display a success message if the cleanup is successful
#           5. Display an error message if the cleanup fails, but continue the program execution
#
def cleanup_files():
    try:
        st.session_state.last_uploaded_files = []
        st.session_state.detect_annotations = {}
        st.session_state.infer_correct = False
        st.session_state.has_infer_result = False
        st.session_state.name_mapping_table = []
        st.session_state.completed_files = []
        shutil.rmtree("inputFile", ignore_errors=True)
        shutil.rmtree("outputFile", ignore_errors=True)
        shutil.rmtree("log", ignore_errors=True)
        os.remove("output_files.zip")
        os.remove("log_files.zip")
        st.success(lang.get("clear_success"))

    except OSError as e:
        st.error(lang.get("clear_fail").format(e=e))
        pass

#
#   Find uuid name function:
#       parameters:
#           name: the name of the file
#           name_mapping_table: a list of tuples containing old and new names
#       function:
#           1. Find the new name of the file in the name_mapping_table
#           2. Return the new name if it is found, otherwise return None
#   
def find_uuid_name(name, name_mapping_table):
    for old_name, new_name in name_mapping_table:
        if name == old_name:
            return new_name
    return None

#
#   Change name to uuid function:
#       parameters:
#           file_name: the name of the file
#           name_mapping_table: a list of tuples containing old and new names
#       function:
#           1. Change the name of the file to a uuid name if the file is not in the name_mapping_table
#           2. Return the new name and the updated name_mapping_table
#
def change_name_to_uuid(file_name, name_mapping_table):
    finding_result = find_uuid_name(file_name, name_mapping_table)
    if finding_result is None:
        new_name = str(uuid.uuid4()) + '.' + file_name.split('.')[-1]
        name_mapping_table.append((file_name, new_name))
        return new_name, name_mapping_table
    else:
        return finding_result, name_mapping_table

#
#   Recover function:
#       parameters:
#           name: the new name of the file
#           name_mapping_table: a list of tuples containing old and new names
#           type: the type of file, either "video" or "photo"
#       function:
#           1. Recover the original name of the file
#           2. Rename the file back to its original name
#           3. Return the old name and the new output path
#
def  recover_name(name, name_mapping_table, type):
    for old_name, new_name in name_mapping_table:
        if name == new_name:
            if type == "video":
                os.rename(f"outputFile/{new_name.split('.')[0]}", f"outputFile/{old_name.split('.')[0]}")
                os.rename(f"outputFile/{old_name.split('.')[0]}/{new_name}", f"outputFile/{old_name.split('.')[0]}/{old_name}")
                new_output_path = f"outputFile/{old_name.split('.')[0]}/{old_name}"
            else:
                os.rename(f"outputFile/photo/{new_name}", f"outputFile/photo/{old_name}")
                new_output_path = f"outputFile/photo/{old_name}"
            return old_name, new_output_path
            
#
# Infer function : 
#    parameters: 
#        args:  paramerters for model initialize, including the path for 
#            input and output file and type of data
#        model: use for inference
#    function:
#        1. Inference the data from user input 
#        2. The interrupt button to stop the inference
#        3. real time inference strealit: 0.5(s), cv2: 0.01(s) 
#
frame_queue = queue.Queue(maxsize=10)
result_queue = queue.Queue(maxsize=10)
ui_queue = queue.Queue(maxsize=3)

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

def writer_thread(result_queue, output_video, draw, total_frames, progress_bar, frame_placeholder, preview_size, lang):
    current_frame = 1
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
        # progress_bar.progress(progress, text=("%.2f" % round(100*current_frame/total_frames, 2) + "%"))  # Update progress bar
        # print(f"Progress: {current_frame} / {total_frames}")
        frame_display = cv2.resize(detect_frame, preview_size, interpolation=cv2.INTER_LINEAR)
        frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
        ui_queue.put((frame_rgb, progress, current_frame))

def infer(args, model, name, format):

    detect_annotation = []
    if st.session_state.infer_correct:
        if args.video: # Add classify video type
            init_time = time.time()
            cap = ffmpegcv.VideoCaptureNV(args.imfile)
            frame_placeholder = st.empty()

            # get the fps, w, h, if the input video
            fps = cap.fps
            width = cap.width
            height = cap.height
            preview_size = (800, height * 800 // width)
            
            # set output video type .mp4
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            output_video = cv2.VideoWriter(os.path.join(args.outputdir, name+".mp4"), fourcc, fps, (width, height))
            
            # Initialize the progress bar with the total number of frames
            total_frames = cap.count
            progress_bar = st.progress(0)  # Progress bar initialized at 0%
            
            # Create a button to interrupt the inference
            interrupt_button = st.button(lang.get("interrupt_inference"), key=name)
            is_interrupted = False
            
            if not cap.isOpened():
                print("cap can not open")
                exit()
            # Diplay inference result in real time
            # frame_placeholder = st.empty()
            ret, frame = cap.read()
            orig_size = torch.tensor([width, height])[None].to(args.device)
            t1 = threading.Thread(target=reader_thread, args=(cap, frame_queue))
            t2 = threading.Thread(target=infer_thread, args=(model, frame_queue, result_queue, orig_size, args.device))
            t3 = threading.Thread(target=writer_thread, args=(result_queue, output_video, draw, total_frames, progress_bar, frame_placeholder, preview_size, lang))

            start_time = time.time()
            t1.start()
            t2.start()
            t3.start()

            while t3.is_alive():
                try:
                    frame_rgb, progress, current_frame = ui_queue.get(timeout=0.1)
                    frame_placeholder.image(frame_rgb, caption=lang.get("on_time_infer_result"), use_container_width=True)
                    progress_bar.progress(progress, text=("%.2f" % round(100*current_frame/total_frames, 2) + "%"))
                except queue.Empty:
                    pass
                # Optionally add a small sleep to avoid busy-waiting
                #time.sleep(0.05)

            t1.join()
            t2.join()
            t3.join()

            # while cap.isOpened():
            #     #t0 = time.perf_counter()
            #     ret, frame = cap.read()
            #     #t1 = time.perf_counter()
            #     if not ret:
            #         break

            #     # Preprocessing
            #     #t2 = time.perf_counter()
            #     frame_resized = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
            #     im_data = (torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0).unsqueeze(0).to(args.device)
            #     #t3 = time.perf_counter()

            #     # Model inference
            #     output = model(im_data, orig_size)
            #     #t4 = time.perf_counter()

            #     # Postprocessing/drawing
            #     labels, boxes, scores = output
            #     detect_frame, box_count = draw([frame], labels, boxes, scores, 0.35)
            #     #t5 = time.perf_counter()

            #     # Streamlit UI update
            #     #frame_pil = Image.fromarray(cv2.cvtColor(cv2.resize(np.array(detect_frame), preview_size, interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB))
            #     #frame_placeholder.image(frame_pil, caption=lang.get("on_time_infer_result"), use_container_width=True, channels = "RGB")
            #     #t6 = time.perf_counter()

            #     # Output video write
            #     output_video.write(detect_frame)
            #     #t7 = time.perf_counter()

            #     # Print timings
            #     #print(f"Read: {t1-t0:.3f}s, Pre: {t3-t2:.3f}s, Infer: {t4-t3:.3f}s, Draw: {t5-t4:.3f}s, UI: {t6-t5:.3f}s, Write: {t7-t6:.3f}s")

            #     # Update the progress bar
            #     #current_frame += 1
            #     #progress = current_frame / total_frames
            #     #progress_bar.progress(progress, text=("%.2f" % round(100*current_frame/total_frames, 2) + "%"))  # Update progress bar
            #     #print(f"Progress: {current_frame} / {total_frames}")

            #     # Collect the frame that is detected
            #     if  box_count > 0:
            #         detect_annotation.append(current_frame)
            cap.release()
            output_video.release()
            end_time = time.time()
            elapsed_time = end_time - init_time
            infer_time = end_time - start_time
            st.info(lang.get("fps").format(fps=round(total_frames/elapsed_time, 2)) + ", " + lang.get("inference_time").format(infer_time=round(infer_time, 2)) + ", " + lang.get("elapsed_time").format(elapsed_time=round(elapsed_time, 2)))

        else:
            start_time = time.time()
            is_interrupted = False
            img = cv2.imread(os.path.join(args.imfile))
            photo_name = args.imfile.split('.')[0].split('/')[-1]
            os.makedirs(args.outputdir, exist_ok=True)
            new_path = args.outputdir
            
            image_resized = cv2.cvtColor(cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
            im_data = (torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0).unsqueeze(0).to(args.device)
            orig_size = torch.tensor([img.shape[1], img.shape[0]])[None].to(args.device)
        
            output = model(im_data, orig_size)
            labels, boxes, scores = output
            detect_frame, box_count = draw([img], labels, boxes, scores, 0.35)
            cv2.imwrite(os.path.join(new_path,f"{photo_name}.{format}"),detect_frame)
            st.image(detect_frame, channels="BGR")
            end_time = time.time()
            elapsed_time = end_time - start_time
            st.info(lang.get("inference_time").format(infer_time=round(elapsed_time, 2)))

        # close all the windows
        # cv2.destroyAllWindows()
        if is_interrupted:
            st.info(lang.get("inference_stopped"))
        else:
            st.success(lang.get("inference_completed"))
        return detect_annotation

#   Make log function:
#       parameters:
#           detect_annotation: the frame number that is detected
#           fps: the frame per second of the video
#           file_name: the name of the file
#       function:
#           1. Create a log file to save the frame number that is detected
#           2. Return the path of the log file
#        
def make_log(detect_annotation, fps, file_name):
    os.makedirs("log", exist_ok=True)
    log_path = os.path.join("log", file_name + '.txt')
    with open(log_path, 'w') as f:
        f.write(lang.get("detected_people"))
        for i in detect_annotation:
            f.write(f"{i/fps:.2f}sec\n")
            
    return log_path

if __name__ == '__main__':
    # Select language
    windll = ctypes.windll.kernel32
    local_lang = locale.windows_locale[ windll.GetUserDefaultUILanguage() ]
    lang_dicts = {"zh_TW": 0 , "en_US": 1}

    selected_language = st.sidebar.selectbox("Select Language", ["zh", "en"], index=lang_dicts.get(local_lang, 1))
    lang = load_language(selected_language)
    st.sidebar.title(lang.get("sidebar_title"))
    page = st.sidebar.selectbox(lang.get("choose_method"), (lang.get("model_system"), lang.get("RTMP_title")))
    if page == "模型偵測系統" or page == "Model System":
        main()
    else:
        stream(lang) 
    
