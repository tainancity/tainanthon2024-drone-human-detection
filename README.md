# tainanthon2024-drone-human-detection
2024台南黑客松-無人機搜救自動辨識
本專案使用 RTDETR 模型進行無人機辨識，支持圖片和影片的上傳。

## 功能

- 上傳圖片或影片
- 使用 RTDETR 模型進行無人機辨識
- 顯示原始圖片或影片
- 顯示預測結果

## 安裝

1. 複製此儲存庫到本地端：

    ```bash
    git clone https://github.com/SeisoNeko/Drone_Human_Detect_ui
    ```

2. 進入專案目錄：

    ```bash
    cd Drone_Human_Detect_ui
    ```

3. 建立並啟動虛擬環境(推薦)：

    ```bash
    conda create -n yourProjectName python=3.8  #3.8以上
    conda activate yourProjectName  
    ```

4. 安裝所需的套件：

    **Windows:**
    ```bash
    install.bat
    ```

5. 放入model  
    請致 rtdetr/weights/ 放入合適的model pth檔  
    並到 rtdetr\tools\infer.py 第256行 輸入model路徑

## 使用方法

1. 啟動 Streamlit 應用：

    ```bash
    streamlit run main.py --server.maxUploadSize 10000
    ```

2. 或者 執行run.bat
    ```bash
    run.bat
    ```

3. 在瀏覽器中打開 `http://localhost:8501`，上傳圖片或影片進行無人機辨識。

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

## 貢獻

歡迎提交問題和請求合併。如果您想貢獻代碼，創建一個 Pull Request。

## 備註
- 本專案使用cuda11.8版本，建議先行安裝適合於本地的cuda tool kit

## 授權

此專案使用 MIT 授權。詳情請參閱 [LICENSE](LICENSE) 文件。