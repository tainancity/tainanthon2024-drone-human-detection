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
from rtdetr.tools.infer import InitArgs, draw, initModel
from anomalyDET import anomaly_main

video_format = ['mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI']
image_format = ['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG']

def main():
    st.title('無人機人員偵測系統')
    
    # upload the file
    uploaded_files = st.file_uploader('上傳圖片/影像', type=['mp4', 'mov', 'avi', 'png', 'jpg', 'jpeg'], accept_multiple_files=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Video_Type = []
    if uploaded_files is not None:
        # 重置 session state 變數
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
        name_mapping_table = st.session_state.name_mapping_table

        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            file_extension = file_name.split('.')[-1]
            
            # classify the input file is video or image
            if file_extension in video_format:
                Video_Type.append(True)     # 代表輸入類型為影片檔
            elif file_extension in image_format:
                Video_Type.append(False)
            else:
                st.warning(f"檔案 {file_name} 格式不支援！")
                continue

            if file_name not in st.session_state.last_uploaded_files:
                st.session_state.detect_annotations[file_name] = None
                st.session_state.last_uploaded_files.append(file_name)

            upload_success = st.success(f"檔案 {file_name} 已成功上傳！")
            if file_extension in image_format:
                st.image(uploaded_file)
            elif file_extension in video_format:
                st.video(uploaded_file)
            else:
                st.warning(f"檔案 {file_name} 格式不支援！")
                continue

            # create dir of to save the input file and inference outcome
            base_name = file_name.split('.')[0]
            uuid_name, name_mapping_table = change_name_to_uuid(file_name, name_mapping_table)

            if Video_Type[-1]:  # 影片檔
                input_dir = os.path.join("inputFile", uuid_name.split('.')[0])
            else:  # 圖片檔
                input_dir = os.path.join("inputFile", "photo")
            output_dir = os.path.join("outputFile", uuid_name.split('.')[0])
            
            try:
                os.makedirs(input_dir, exist_ok=True)
                if Video_Type[-1]:  # 圖片需要輸出目錄
                    os.makedirs(output_dir, exist_ok=True)
            except Exception as e:
                st.error(f"目錄創建失敗：{e}")
                continue


            # copy the video to inputFile
            save_path = os.path.join(input_dir, uuid_name)
            try:
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                fps = cv2.VideoCapture(save_path).get(cv2.CAP_PROP_FPS)
            except Exception as e:
                st.error(f"文件保存失敗：{e}")
                
            # close the success message    
            upload_success.empty()
            
            save_success = st.success(f"檔案已儲存至 {save_path}")
            save_success.empty()
        
        # 顯示「開始推理」按鈕
        if st.session_state.infer_correct == False and st.session_state.last_uploaded_files != []:
            st.session_state.infer_correct = st.button("開始推理")

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
                        st.video(new_output_path)
                        log_path = make_log(st.session_state.detect_annotations[file_name], fps, original_name.split('.')[0])
                        st.success(f"偵測結果已儲存至 {log_path}")
            st.session_state.infer_correct = False
            st.session_state.has_infer_result = True

        if st.session_state.has_infer_result:
            # 提供下載壓縮檔案的按鈕
            zip_path = zip_output_files()
            with open(zip_path, "rb") as f:
                st.download_button('下載所有預測檔案', f, 'output_files.zip')
            
            # 提供下載log檔案的按鈕
            log_zip_path = zip_log_files()
            with open(log_zip_path, "rb") as f:
                st.download_button('下載所有log檔案', f, 'log_files.zip')

            # 清理掉臨時檔案
    st.button("清理臨時檔案", on_click=lambda: cleanup_files())

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
        shutil.rmtree("inputFile", ignore_errors=True)
        shutil.rmtree("outputFile", ignore_errors=True)
        shutil.rmtree("log", ignore_errors=True)
        os.remove("output_files.zip")
        os.remove("log_files.zip")
        st.success("成功清理所有臨時檔案.")

    except OSError as e:
        st.error(f"清除檔案失敗: {e}")
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
def infer(args, model, name, format):

    detect_annotation = []
    if st.session_state.infer_correct:
        if args.video: # Add classify video type
            cap = cv2.VideoCapture(args.imfile)
            
            # get the fps, w, h, if the input video
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # set output video type .mp4
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            output_video = cv2.VideoWriter(os.path.join(args.outputdir, name+".mp4"), fourcc, fps, (width, height))
            
            # Initialize the progress bar with the total number of frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress_bar = st.progress(0)  # Progress bar initialized at 0%
            
            # Create a button to interrupt the inference
            interrupt_button = st.button('中斷推理', key=name)
            is_interrupted = False
            
            if not cap.isOpened():
                print("cap can not open")
                exit()
            # Diplay inference result in real time
            # frame_placeholder = st.empty()
            while cap.isOpened():
                
                ret, frame = cap.read()
                if not ret:
                    print("Frame end or can not read frame")
                    break
                if interrupt_button:
                    st.session_state.infer_correct = False
                    is_interrupted = True
                    st.warning("推理已中斷！")
                    break  # Break the loop to stop the inference
                    
                # change the graph type from bgr to rgb 
                im_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                w, h = im_pil.size
                orig_size = torch.tensor([w, h])[None].to(args.device)
            
                # Resize the graph and change to tensor type to inference
                transforms = T.Compose([
                    T.Resize((640, 640)),  
                    T.ToTensor(),
                ])
                im_data = transforms(im_pil)[None].to(args.device)
                    
                output = model(im_data, orig_size)
                labels, boxes, scores = output
                    
                detect_frame, box_count = draw([im_pil], labels, boxes, scores, 0.35)
                frame_out = cv2.cvtColor(np.array(detect_frame), cv2.COLOR_RGB2BGR)
                # Display inference result
                cv2.imshow("Real time Inference", cv2.resize(frame_out, (800, 600)))
                cv2.waitKey(1)
                
                output_video.write(frame_out)
                
                # Update the progress bar
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                progress = current_frame / total_frames
                progress_bar.progress(progress)  # Update progress bar
                print(f"Progress: {current_frame} / {total_frames}")

                # Collect the frame that is detected
                if  box_count > 0:
                    detect_annotation.append(current_frame)
            cap.release()
            output_video.release()

        else:
            is_interrupted = False
            img = cv2.imread(os.path.join(args.imfile))
            photo_name = args.imfile.split('.')[0].split('/')[-1]
            os.makedirs(args.outputdir, exist_ok=True)
            new_path = args.outputdir
            
            im_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            w, h = im_pil.size
            orig_size = torch.tensor([w, h])[None].to(args.device)
        
            # Resize the graph and change to tensor type to inference
            transforms = T.Compose([
                T.Resize((640, 640)),  
                T.ToTensor(),
            ])
            im_data = transforms(im_pil)[None].to(args.device)
            output = model(im_data, orig_size)
            labels, boxes, scores = output
            detect_frame, box_count = draw([im_pil], labels, boxes, scores, 0.35)
            frame_out = cv2.cvtColor(np.array(detect_frame), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(new_path,f"{photo_name}.{format}"),frame_out)
            st.image(frame_out, channels="BGR")

        # close all the windows
        cv2.destroyAllWindows()
        if is_interrupted:
            st.info("推理過程已停止。")
        else:
            st.success("推理完成")
        return detect_annotation

#
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
        f.write("偵測到的人員在以下秒數：\n")
        for i in detect_annotation:
            f.write(f"{i/fps:.2f}秒\n")
            
    return log_path

if __name__ == '__main__':
    st.sidebar.title('選用預測方法')
    page = st.sidebar.selectbox("選擇預測方法", ("模型偵測系統", "異常偵測系統"))
    if page == "模型偵測系統":
        main()
    else:
        anomaly_main()
    