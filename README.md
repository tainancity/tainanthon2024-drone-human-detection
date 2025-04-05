# tainanthon2024-drone-human-detection
2024台南黑客松-無人機搜救自動辨識

本專案使用 RTDETR 模型進行無人機影像辨識，支援圖片和影片格式。

## 功能

- 上傳圖片或影片
- 使用 RTDETR 模型或是異常偵測(實驗性質)進行無人機辨識
- 顯示原始圖片或影片
- 顯示與下載預測結果

## 安裝

1. 複製整個Repo到本地：

    ```bash
    git clone https://github.com/SeisoNeko/Drone_Human_Detect_ui
    ```

2. 進入專案根目錄：

    ```bash
    cd Drone_Human_Detect_ui
    ```

3. 建立並啟動虛擬環境(推薦)：

    ```bash
    conda create -n yourEnvName python=3.8  #3.8以上
    conda activate yourEnvName  
    ```

5. 安裝所需的套件：

    **Windows:**
    ```bash
    install.bat
    ```

6. 放入model  
    請將合適的model weight檔放入 rtdetr/weights/  
    並到 rtdetr/tools/infer.py 第256行 修改model路徑

## 使用方法

1. 啟動 Streamlit 應用：
   
    使用 v1 模型
    ```bash
    streamlit run main.py --server.maxUploadSize 10000
    ```

    使用 v2 模型
    ```bash
    streamlit run main_v2.py --server.maxUploadSize 10000
    ```

3. 或者 直接執行`run.bat`

4. 在瀏覽器中打開 `http://localhost:8501`(如果沒有自動開啟的話)，上傳圖片或影片進行影像辨識。

## 專案結構

- `main.py`：主應用程式碼
- `requirements.txt`：所需的 Python 套件
- `weights/`：存放 RTDETR 模型(checkpoints)的目錄
- `outputFile/`：存放預測結果的目錄

## 依賴項目

- Python 3.8+
- Streamlit
- ultralytics
- PIL (Pillow)
- cv2
- CUDA Toolkit 11.8

## 授權

此專案使用 MIT 授權。詳情請參閱 [LICENSE](LICENSE) 文件。
