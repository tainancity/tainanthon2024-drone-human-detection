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
     
    args = InitArgs(imfile=None, video=False, outputdir=None, device=device)
    model = initModel(args)

    if not cap.isOpened():
        st.error(lang.get("fail_open_rtmp"))
    else:
        st.success(lang.get("success_open_rtmp"))
        frame_placeholder = st.empty()
        infer_frame_placeholder = st.empty()
        stop = st.button("Stop", key="stop_button")

        current_time = time.time()
        new_time = time.time()

        fps_info = st.empty()
        fps = 0.0

        while cap.isOpened() and not stop:
            fps_info.info(f"FPS: {fps:.2f}")
            current_time = new_time
            ret, frame = cap.read()
            if not ret:
                st.warning(lang.get("fail_read_frame"))
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame, channels="RGB")

            infer_frame = infer_stream(args, frame, model)
            infer_frame = cv2.cvtColor(infer_frame, cv2.COLOR_BGR2RGB)
            infer_frame_placeholder.image(infer_frame, channels="RGB")
            new_time = time.time()
            fps = 1 / (new_time - current_time)
            fps_info.empty() 
        cap.release()

def infer_stream(args, frame, model):
    # Placeholder for inference logic
    # Replace with actual inference code
    im_pil = Image.fromarray(frame)
    w,h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(args.device)

    transforms = T.Compose([ 
                T.Resize((640, 640)),  
                T.ToTensor(),
            ])
    
    im_data = transforms(im_pil)[None].to(args.device)
    output = model(im_data, orig_size)
    labels, boxes, scores = output
    detect_frame, box_count = draw([im_pil], labels, boxes, scores, 0.35)
    frame_out = cv2.cvtColor(np.array(detect_frame), cv2.COLOR_RGB2BGR)

    return frame_out

if __name__ == "__main__":
    stream()