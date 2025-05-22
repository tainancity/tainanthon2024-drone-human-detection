document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const uploadStatus = document.getElementById('uploadStatus');
    
    const fileSelectionStatus = document.getElementById('fileSelectionStatus'); // 新增：用於簡潔的檔案選擇提示

    const selectedPreviewsContainer = document.getElementById('selectedPreviewsContainer'); // 新增：選擇檔案後預覽的容器
    const selectedPreviews = document.getElementById('selectedPreviews'); // 新增：選擇檔案後預覽的內容區

    const uploadedFilesDisplay = document.getElementById('uploadedFilesDisplay');
    const uploadedPreviews = document.getElementById('uploadedPreviews'); // 已上傳檔案的預覽內容區

    const inferBtn = document.getElementById('inferBtn');
    const inferenceStatus = document.getElementById('inferenceStatus');
    const inferenceResults = document.getElementById('inferenceResults');
    const downloadOutputZipBtn = document.getElementById('downloadOutputZipBtn');
    const downloadLogZipBtn = document.getElementById('downloadLogZipBtn');
    const cleanupBtn = document.getElementById('cleanupBtn');
    const cleanupStatus = document.getElementById('cleanupStatus');

    let uploadedFilesInfo = []; // 儲存已上傳的檔案資訊（包括後端返回的 uuid_name 等）

    // 輔助函數：判斷檔案是否為影片
    function isVideoFile(fileName) {
        const videoFormats = ['mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI'];
        const extension = fileName.split('.').pop().toLowerCase();
        return videoFormats.includes(extension);
    }

    // 輔助函數：判斷檔案是否為圖片
    function isImageFile(fileName) {
        const imageFormats = ['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG'];
        const extension = fileName.split('.').pop().toLowerCase();
        return imageFormats.includes(extension);
    }

    // 顯示狀態訊息的輔助函數
    function displayStatus(element, message, type) {
        element.textContent = message;
        element.className = `status-message ${type}`;
    }

    // 處理檔案選擇並顯示本地預覽 [解決問題 1 和 2 的「選擇檔案後預覽」]
    fileInput.addEventListener('change', (event) => {
        // 清空之前的狀態和預覽
        uploadStatus.innerHTML = '';
        inferenceResults.innerHTML = '';
        inferenceStatus.innerHTML = '';
        uploadedPreviews.innerHTML = ''; // 清空上傳預覽
        selectedPreviews.innerHTML = ''; // 清空選擇檔案預覽
        fileSelectionStatus.textContent = ''; // 清空文字提示
        
        uploadedFilesInfo = []; // 清空已上傳檔案資訊
        inferBtn.disabled = true; // 選擇新檔案後禁用推理按鈕
        
        const files = event.target.files;
        if (files.length === 0) {
            displayStatus(fileSelectionStatus, '請選擇檔案...', 'info');
            selectedPreviewsContainer.style.display = 'none'; // 隱藏預覽區塊
            return;
        }

        let fileNames = Array.from(files).map(file => file.name).join(', ');
        displayStatus(fileSelectionStatus, `已選擇檔案: ${fileNames}`, 'info');
        selectedPreviewsContainer.style.display = 'block'; // 顯示預覽區塊

        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const itemDiv = document.createElement('div');
            itemDiv.className = 'media-preview-item';
            itemDiv.innerHTML = `<h4>${file.name}</h4>`;

            const reader = new FileReader();
            reader.onload = (e) => {
                const fileType = file.type;
                if (fileType.startsWith('image/')) {
                    const img = document.createElement('img');
                    img.src = e.target.result; // Data URL for image
                    img.alt = file.name;
                    itemDiv.appendChild(img);
                } else if (fileType.startsWith('video/')) {
                    const video = document.createElement('video');
                    video.controls = true;
                    video.src = e.target.result; // Data URL for video
                    video.muted = true; // 預覽時通常靜音
                    video.loop = false; // 預覽時循環播放
                    video.autoplay = false; // 自動播放
                    itemDiv.appendChild(video);
                } else {
                    itemDiv.innerHTML += `<p>不支持的預覽類型: ${file.name}</p>`;
                }
                selectedPreviews.appendChild(itemDiv);
            };
            reader.readAsDataURL(file); // 讀取檔案為 Data URL
        }
    });

    // 處理上傳按鈕點擊
    uploadBtn.addEventListener('click', async () => {
        const files = fileInput.files;
        if (files.length === 0) {
            displayStatus(uploadStatus, '請選擇要上傳的檔案。', 'error');
            return;
        }

        displayStatus(uploadStatus, '檔案上傳中，請稍候...', 'info');
        uploadedPreviews.innerHTML = ''; // 清空上一個上傳的預覽
        selectedPreviews.innerHTML = ''; // 清空選擇檔案預覽 (上傳後這些預覽已無用)
        selectedPreviewsContainer.style.display = 'none'; // 隱藏選擇檔案預覽區塊

        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
            formData.append('files[]', files[i]);
        }

        try {
            const response = await fetch('http://127.0.0.1:5000/upload', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();

            if (data.success) {
                uploadedFilesInfo = data.results; // 更新已上傳檔案資訊
                displayStatus(uploadStatus, '檔案上傳完成！', 'success');
                
                // 顯示上傳檔案的預覽 [解決問題 2 的「上傳後預覽」]
                uploadedFilesInfo.forEach(file => {
                    const itemDiv = document.createElement('div');
                    itemDiv.className = 'media-preview-item';
                    itemDiv.innerHTML = `<h4>${file.original_name}</h4>`;
                    
                    if (file.success) {
                        const fileExtension = file.original_name.split('.').pop().toLowerCase();
                        let mediaElement;
                        // 構建後端提供原始檔案的 URL
                        // 後端需要提供 `/static_input/` 路由來服務 `inputFile`
                        const staticInputUrl = `http://127.0.0.1:5000/static_input/`;
                        const relativePath = file.is_video ? `${file.uuid_name.split('.')[0]}/${file.uuid_name}` : `photo/${file.uuid_name}`;
                        const displayPath = staticInputUrl + relativePath;

                        if (file.is_video) {
                            mediaElement = document.createElement('video');
                            mediaElement.controls = true;
                            mediaElement.muted = true;
                            mediaElement.loop = false;
                            mediaElement.autoplay = false;
                        } else if (isImageFile(file.original_name)) {
                            mediaElement = document.createElement('img');
                        } else {
                            itemDiv.innerHTML += `<p style="color: orange;">不支持的預覽類型: ${file.original_name}</p>`;
                            uploadedPreviews.appendChild(itemDiv);
                            return; // 跳過當前檔案的媒體元素創建
                        }
                        mediaElement.src = displayPath;
                        mediaElement.alt = file.original_name;
                        itemDiv.appendChild(mediaElement);
                        itemDiv.innerHTML += `<p style="color: green;">上傳成功。</p>`;
                    } else {
                        itemDiv.innerHTML += `<p style="color: red;">上傳失敗: ${file.message}</p>`;
                    }
                    uploadedPreviews.appendChild(itemDiv);
                });
                inferBtn.disabled = false; // 上傳成功後啟用推理按鈕

            } else {
                displayStatus(uploadStatus, `上傳失敗: ${data.message}`, 'error');
            }
        } catch (error) {
            displayStatus(uploadStatus, `上傳錯誤: ${error.message}`, 'error');
            console.error('Upload error:', error);
        }
    });

    // 處理推理按鈕點擊
    inferBtn.addEventListener('click', async () => {
        if (uploadedFilesInfo.length === 0) {
            displayStatus(inferenceStatus, '請先上傳檔案。', 'error');
            return;
        }

        displayStatus(inferenceStatus, '模型推理中，這可能需要一些時間...', 'info');
        inferenceResults.innerHTML = ''; // 清空之前的結果
        downloadOutputZipBtn.disabled = true;
        downloadLogZipBtn.disabled = true;

        try {
            const response = await fetch('http://127.0.0.1:5000/infer', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({})
            });
            const data = await response.json();

            if (data.success) {
                displayStatus(inferenceStatus, '模型推理完成！', 'success');
                downloadOutputZipBtn.disabled = false;
                downloadLogZipBtn.disabled = false;
                
                data.results.forEach(result => {
                    const itemDiv = document.createElement('div');
                    itemDiv.className = 'inference-item';
                    itemDiv.innerHTML = `<h3>${result.original_name}</h3>`;
                    
                    if (result.success) {
                        itemDiv.innerHTML += `<p style="color: green;">${result.message}</p>`;
                        if (result.output_path) {
                            // 構建後端提供處理結果檔案的 URL
                            // Flask 後端在 `app.static_url_path = '/static_output'` 時
                            // 會將 `outputFile` 目錄下的檔案映射到 `/static_output/` URL 前綴
                            const staticOutputUrl = `http://127.0.0.1:5000/static_output/`;
                            // output_path 是 Flask 後端 `recover_name` 函數返回的絕對或相對路徑，
                            // 需要調整為相對於 `outputFile` 目錄的 URL 路徑
                            // 例如：`outputFile/folder/file.mp4` -> `folder/file.mp4`
                            const relativeOutputPath = result.output_path.replace(/\\/g, '/').replace('outputFile/', '');
                            const displayPath = staticOutputUrl + relativeOutputPath;
                            
                            let mediaElement;
                            if (result.is_video) {
                                mediaElement = document.createElement('video');
                                mediaElement.controls = true;
                                mediaElement.autoplay = true;
                                mediaElement.loop = true;
                                mediaElement.muted = true;
                            } else {
                                mediaElement = document.createElement('img');
                            }
                            mediaElement.src = displayPath;
                            mediaElement.alt = "Inferred Result";
                            itemDiv.appendChild(mediaElement);
                        }
                        if (result.log_path) {
                            itemDiv.innerHTML += `<p>日誌檔路徑: ${result.log_path.replace(/\\/g, '/')}</p>`;
                        }
                    } else {
                        itemDiv.innerHTML += `<p style="color: red;">推理失敗: ${result.message}</p>`;
                    }
                    inferenceResults.appendChild(itemDiv);
                });

            } else {
                displayStatus(inferenceStatus, `推理失敗: ${data.message}`, 'error');
            }
        } catch (error) {
            displayStatus(inferenceStatus, `推理錯誤: ${error.message}`, 'error');
            console.error('Inference error:', error);
        }
    });

    // 下載壓縮檔
    downloadOutputZipBtn.addEventListener('click', () => {
        window.location.href = 'http://127.0.0.1:5000/download_output_zip';
    });

    downloadLogZipBtn.addEventListener('click', () => {
        window.location.href = 'http://127.0.0.1:5000/download_log_zip';
    });

    // 清除檔案
    cleanupBtn.addEventListener('click', async () => {
        displayStatus(cleanupStatus, '正在清除檔案...', 'info');
        try {
            const response = await fetch('http://127.0.0.1:5000/cleanup', {
                method: 'POST'
            });
            const data = await response.json();
            if (data.success) {
                displayStatus(cleanupStatus, data.message, 'success');
                // 清空前端顯示狀態
                uploadedFilesInfo = [];
                fileSelectionStatus.textContent = '請選擇檔案...'; // 重置選擇檔案提示
                selectedPreviews.innerHTML = ''; // 清空選擇檔案預覽
                selectedPreviewsContainer.style.display = 'none'; // 隱藏預覽區塊
                uploadedPreviews.innerHTML = ''; // 清空上傳預覽
                inferenceResults.innerHTML = '';
                inferenceStatus.innerHTML = '';
                uploadStatus.innerHTML = '';
                downloadOutputZipBtn.disabled = true;
                downloadLogZipBtn.disabled = true;
                fileInput.value = ''; // 清空檔案選擇框
                inferBtn.disabled = true; // 清除後禁用推理按鈕
            } else {
                displayStatus(cleanupStatus, `清除失敗: ${data.message}`, 'error');
            }
        } catch (error) {
            displayStatus(cleanupStatus, `清除錯誤: ${error.message}`, 'error');
            console.error('Cleanup error:', error);
        }
    });

    // 初始狀態
    displayStatus(fileSelectionStatus, '請選擇檔案...', 'info');
    selectedPreviewsContainer.style.display = 'none'; // 初始隱藏預覽區塊
});