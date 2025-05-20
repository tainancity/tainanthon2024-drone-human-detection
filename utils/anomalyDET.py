import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time

input_tempfile_path = []
output_tempfile_path = []

# 定義增強與降低亮度的函數
def enhance_and_downgrade_colors(image, target_colors, downgrade_colors, tolerance=50, downgrade_factor=0.5):
    """增強目標顏色，降低指定顏色亮度"""
    enhanced_image = image.copy()
    target_mask_total = np.zeros(image.shape[:2], dtype=np.uint8)
    downgrade_mask_total = np.zeros(image.shape[:2], dtype=np.uint8)

    # 增強目標顏色
    for target_rgb in target_colors:
        lower_bound = np.clip(np.array(target_rgb) - tolerance, 0, 255)
        upper_bound = np.clip(np.array(target_rgb) + tolerance, 0, 255)
        mask = cv2.inRange(image, lower_bound, upper_bound)
        target_mask_total = cv2.bitwise_or(target_mask_total, mask)  # 疊加所有目標顏色
        enhanced_image[mask > 0] = np.clip(enhanced_image[mask > 0] * 2, 0, 255).astype(np.uint8)

    # 降低指定顏色亮度
    for downgrade_rgb in downgrade_colors:
        lower_bound = np.clip(np.array(downgrade_rgb) - tolerance, 0, 255)
        upper_bound = np.clip(np.array(downgrade_rgb) + tolerance, 0, 255)
        mask = cv2.inRange(image, lower_bound, upper_bound)
        downgrade_mask_total = cv2.bitwise_or(downgrade_mask_total, mask)  # 疊加所有降低亮度顏色
        enhanced_image[mask > 0] = np.clip(enhanced_image[mask > 0] * downgrade_factor, 0, 255).astype(np.uint8)

    return enhanced_image

# 定義RX檢測函式
def calculate_rx(image, mean, inv_cov):
    h, w, d = image.shape
    pixels = image.reshape(-1, d).astype(float)
    diff = pixels - mean
    rx_scores = np.einsum('ij,jk,ik->i', diff, inv_cov, diff)
    rx_scores = rx_scores.reshape(h, w)
    return rx_scores

# 處理影片函式
def process_video(input_video_path, output_video_path, target_colors, downgrade_colors, tolerance, rx_threshold, downgrade_factor):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        st.error(f"Unable to open video {input_video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 總幀數

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height * 2))

    # 初始化進度條
    progress_bar = st.progress(0)

    frame_placeholder = st.empty()
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 生成處理後圖像
        processed_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_image = enhance_and_downgrade_colors(processed_image, target_colors, downgrade_colors, tolerance, downgrade_factor)

        # 呼叫並計算RX檢測
        pixels = processed_image.reshape(-1, processed_image.shape[2]).astype(float)
        mean = np.mean(pixels, axis=0)
        cov = np.cov(pixels, rowvar=False)
        inv_cov = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))

        rx_scores = calculate_rx(processed_image, mean, inv_cov)
        anomaly_map = (rx_scores > rx_threshold).astype(np.uint8) * 255
        anomaly_map_color = cv2.cvtColor(anomaly_map, cv2.COLOR_GRAY2BGR)

        # 創建一個合併的顯示圖 上原始圖像 下黑白異常檢測圖
        combined_display = np.zeros((frame_height * 2, frame_width, 3), dtype=np.uint8)
        combined_display[:frame_height, :] = frame  # 上 原影片
        combined_display[frame_height:, :] = anomaly_map_color  # 下 黑白異常圖

        frame_placeholder.image(combined_display, channels="BGR")
        # 儲存輸出影片
        out.write(combined_display)

        frame_count += 1
        progress_bar.progress(frame_count / total_frames)  # 更新進度條

    cap.release()
    out.release()

def anomaly_main():

    # Streamlit ui
    st.title("異物檢測系統")

    # 上傳影片
    uploaded_video = st.file_uploader("上傳影片", type=["mp4", "avi", "mov", "MP4", "AVI", "MOV"])
    if uploaded_video:
        file_name = uploaded_video.name.split(".")[0] # 取得檔案名稱
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as input_temp:
            input_temp.write(uploaded_video.read())
            input_video_path = input_temp.name
            input_tempfile_path.append(input_video_path)
            print(input_video_path)

        st.video(input_video_path)
        # 選擇目標顏色
        color_hex = st.color_picker("選擇想要增強的顏色", "#FFCC99")  # 初始值為皮膚色
        selected_color = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

        # 選擇降階顏色
        downgrade_color_hex = st.color_picker("選擇想要削弱的顏色", "#808080")  # 初始值為灰色
        downgrade_color = tuple(int(downgrade_color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

        # 顯示選定顏色的rgb值
        st.write(f"選擇的增強顏色 (RGB): {selected_color}")
        st.write(f"選擇的削弱顏色 (RGB): {downgrade_color}")

        # 設置參數
        rx_threshold = st.slider("RX Detection 強度", min_value=1, max_value=100, value=40, help='數值越高，檢測越嚴格，偵測到的異常越少，雜訊越少')
        tolerance = st.slider("顏色差異容許度", min_value=1, max_value=100, value=50, help='數值越高，容許的顏色差異越大')
        downgrade_factor = st.slider("削弱顏色強度", min_value=0.0, max_value=1.0, value=0.5, help='數值越低，過濾顏色越多')

        # 保存輸出影片的選項
        save_output = st.checkbox("保存輸出影片")

        # 點擊處理按紐
        if st.button("開始偵測"):
            output_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            output_video_path = output_temp.name
            output_tempfile_path.append(output_video_path)
            print(output_video_path)

            target_colors = [selected_color]
            downgrade_colors = [downgrade_color]
            process_video(input_video_path, output_video_path, target_colors, downgrade_colors, tolerance, rx_threshold, downgrade_factor)

            # 顯示處理後的影片
            st.success("偵測完成!")
            st.video(output_video_path)

            # 提供下載影片
            if save_output:
                st.write("點擊按鈕以下載影片:")
                with open(output_video_path, "rb") as file:
                    btn = st.download_button(
                        label="Download Video",
                        data=file,
                        file_name= file_name +"_processed.mp4",
                        mime="video/mp4"
                    )

            # 清理掉臨時檔案
            if not save_output:
                st.button("清理臨時檔案", on_click=lambda: cleanup_files())

def cleanup_files():
    time.sleep(2)  

    for path in input_tempfile_path:
        try:
            os.remove(path)
            st.success("成功清理輸入影片檔案.")
        except OSError as e:
            st.error(f"清除輸入影片失敗: {e}")

    for path in output_tempfile_path:
        try:
            os.remove(path)
            st.success("成功清理輸出影片檔案.")
        except OSError as e:
            st.error(f"清除輸出影片失敗: {e}")

    input_tempfile_path.clear()
    output_tempfile_path.clear()

if __name__ == '__main__':
    anomaly_main()