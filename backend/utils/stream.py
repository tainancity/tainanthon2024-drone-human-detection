import cv2
import streamlit as st
import torch
import os
import time
import numpy as np
import torchvision.transforms as T
from rtdetrv2.tools.infer import draw, initModel, InitArgs
from PIL import Image
import ffmpegcv
import time
import queue
import threading

frame_queue = queue.Queue(maxsize=10)
result_queue = queue.Queue(maxsize=10)
ui_queue = queue.Queue(maxsize=2)

def reader_thread(cap, frame_queue, frame_placeholder):
    while True:
        ret, frame = cap.read()
        if not ret:
            frame_queue.put(None)
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB")
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

def writer_thread(result_queue, output_video, draw, infer_frame_placeholder, lang):
    while True:
        item = result_queue.get()
        if item is None:
            break
        frame, output = item
        labels, boxes, scores = output
        detect_frame, box_count = draw([frame], labels, boxes, scores, 0.35)
        #output_video.write(detect_frame)
        # Optionally update UI here, but not every frame

        ui_queue.put(detect_frame)

def stream(lang):
    #stream_url = "rtmp://localhost:1935/live/test"
    stream_url = "rtsp://rpi.local:8554/cam"
    #stream_url = "http://192.168.100.101:8889/cam"

    st.title(lang.get("RTMP_title"))

    stream_url = st.text_input("RTMP Stream URL", value=stream_url)

    # device = cv2.cuda.Device(0) if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gstreamer_str = (
        f'rtspsrc protocols=tcp location={stream_url} latency=0 ! '
        'decodebin ! videoconvert ! appsink'
    )
    #cap = cv2.VideoCapture(gstreamer_str, cv2.CAP_GSTREAMER)
    
    cap = ffmpegcv.VideoCaptureStreamRT(stream_url, codec="h264", gpu=0)
    width = cap.width
    height = cap.height
    preview_size = (800, height * 800 // width)
    orig_size = torch.tensor([width, height])[None].to(args.device)
    args = InitArgs(imfile=None, video=False, outputdir=None, device=device)
    model = initModel(args)

    if not cap.isOpened():
        st.error(lang.get("fail_open_rtmp"))
    else:
        st.success(lang.get("success_open_rtmp"))
        frame_placeholder = st.empty()
        infer_frame_placeholder = st.empty()
        stop = st.button("Stop", key="stop_button")

        # set output video type .mp4
        # fourcc = cv2.VideoWriter_fourcc(*'avc1')
        # output_video = cv2.VideoWriter(os.path.join(args.outputdir, name+".mp4"), fourcc, fps, (width, height))
        
        t1 = threading.Thread(target=reader_thread, args=(cap, frame_queue, frame_placeholder))
        t2 = threading.Thread(target=infer_thread, args=(model, frame_queue, result_queue, orig_size, args.device))
        t3 = threading.Thread(target=writer_thread, args=(result_queue, draw, infer_frame_placeholder, preview_size, lang))

        current_time = time.time()
        new_time = time.time()

        fps_info = st.empty()
        fps = 0.0
        t1.start()
        t2.start()
        t3.start()

        while t3.is_alive():
            fps_info.info(f"FPS: {fps:.2f}")
            current_time = new_time
            try:
                frame_rgb = ui_queue.get(timeout=0.1)
                new_time = time.time()
                fps = 1 / (new_time - current_time)
                frame_placeholder.image(frame_rgb, caption=lang.get("on_time_infer_result"), use_container_width=True)
            except queue.Empty:
                pass
            # Optionally add a small sleep to avoid busy-waiting
            #time.sleep(0.05)

        t1.join()
        t2.join()
        t3.join()
        

        # while cap.isOpened() and not stop:
        #     fps_info.info(f"FPS: {fps:.2f}")
        #     current_time = new_time
        #     ret, frame = cap.read()
        #     if not ret:
        #         st.warning(lang.get("fail_read_frame"))
        #         break
        #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #     frame_placeholder.image(frame, channels="RGB")

        #     infer_frame = infer_stream(args, frame, model)
        #     infer_frame = cv2.cvtColor(infer_frame, cv2.COLOR_BGR2RGB)
        #     infer_frame_placeholder.image(infer_frame, channels="RGB")
        #     new_time = time.time()
        #     fps = 1 / (new_time - current_time)
        #     fps_info.empty() 
        cap.release()

if __name__ == "__main__":
    stream()