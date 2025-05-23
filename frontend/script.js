document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const uploadStatus = document.getElementById('uploadStatus');

    const fileSelectionStatus = document.getElementById('fileSelectionStatus');

    const selectedPreviewsContainer = document.getElementById('selectedPreviewsContainer');
    const selectedPreviews = document.getElementById('selectedPreviews');

    const uploadedFilesDisplay = document.getElementById('uploadedFilesDisplay');
    const uploadedPreviews = document.getElementById('uploadedPreviews');

    const inferBtn = document.getElementById('inferBtn');
    const inferenceStatus = document.getElementById('inferenceStatus');
    const inferenceResults = document.getElementById('inferenceResults');
    const downloadOutputZipBtn = document.getElementById('downloadOutputZipBtn');
    const downloadLogZipBtn = document.getElementById('downloadLogZipBtn');
    const cleanupBtn = document.getElementById('cleanupBtn');
    const cleanupStatus = document.getElementById('cleanupStatus');

    let uploadedFilesInfo = [];

    function isVideoFile(fileName) {
        const videoFormats = ['mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI'];
        const extension = fileName.split('.').pop().toLowerCase();
        return videoFormats.includes(extension);
    }

    function isImageFile(fileName) {
        const imageFormats = ['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG'];
        const extension = fileName.split('.').pop().toLowerCase();
        return imageFormats.includes(extension);
    }

    function displayStatus(element, message, type) {
        element.textContent = message;
        element.className = `status-message ${type}`;
    }

    fileInput.addEventListener('change', (event) => {
        uploadStatus.innerHTML = '';
        inferenceResults.innerHTML = '';
        inferenceStatus.innerHTML = '';
        uploadedPreviews.innerHTML = '';
        selectedPreviews.innerHTML = '';
        fileSelectionStatus.textContent = '';

        uploadedFilesInfo = [];
        inferBtn.disabled = true;

        const files = event.target.files;
        if (files.length === 0) {
            displayStatus(fileSelectionStatus, '請選擇檔案...', 'info');
            selectedPreviewsContainer.style.display = 'none';
            return;
        }

        let fileNames = Array.from(files).map(file => file.name).join(', ');
        displayStatus(fileSelectionStatus, `已選擇檔案: ${fileNames}`, 'info');
        selectedPreviewsContainer.style.display = 'block';

        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const itemDiv = document.createElement('div');
            itemDiv.className = 'media-preview-item';
            itemDiv.innerHTML = `<h4>${file.name}</h4>`;

            const reader = new FileReader();
            reader.onload = (e) => {
                if (file.type.startsWith('image/')) {
                    const img = document.createElement('img');
                    img.src = e.target.result;
                    img.alt = file.name;
                    itemDiv.appendChild(img);
                } else if (file.type.startsWith('video/')) {
                    const video = document.createElement('video');
                    video.controls = true;
                    video.src = e.target.result;
                    video.muted = true;
                    video.loop = false;
                    video.autoplay = false;
                    itemDiv.appendChild(video);
                } else {
                    itemDiv.innerHTML += `<p>不支持的預覽類型: ${file.name}</p>`;
                }
                selectedPreviews.appendChild(itemDiv);
            };
            reader.readAsDataURL(file);
        }
    });

    uploadBtn.addEventListener('click', async () => {
        const files = fileInput.files;
        if (files.length === 0) {
            displayStatus(uploadStatus, '請選擇要上傳的檔案。', 'error');
            return;
        }

        displayStatus(uploadStatus, '檔案上傳中，請稍候...', 'info');
        uploadedPreviews.innerHTML = '';
        selectedPreviews.innerHTML = '';
        selectedPreviewsContainer.style.display = 'none';

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
                uploadedFilesInfo = data.results;
                displayStatus(uploadStatus, '檔案上傳完成！', 'success');

                uploadedFilesInfo.forEach(file => {
                    const itemDiv = document.createElement('div');
                    itemDiv.className = 'media-preview-item';
                    itemDiv.innerHTML = `<h4>${file.original_name}</h4>`;

                    if (file.success) {
                        // 構建後端提供原始檔案的 URL
                        const staticInputUrl = `http://127.0.0.1:5000/static_input/`;
                        // 注意：這裡假設 file.uuid_name 已經包含副檔名
                        const relativePath = file.is_video ? `${file.uuid_name.split('.')[0]}/${file.uuid_name}` : `photo/${file.uuid_name}`;
                        const displayPath = staticInputUrl + relativePath;

                        let mediaElement;
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
                            return;
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
                inferBtn.disabled = false;

            } else {
                displayStatus(uploadStatus, `上傳失敗: ${data.message}`, 'error');
            }
        } catch (error) {
            displayStatus(uploadStatus, `上傳錯誤: ${error.message}`, 'error');
            console.error('Upload error:', error);
        }
    });

    inferBtn.addEventListener('click', async () => {
        // ... (省略前面判斷和狀態顯示的程式碼) ...

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
                        if (result.output_path_relative) { // 使用新的鍵名
                            // 直接拼接後端返回的相對路徑。
                            // Flask 的 send_from_directory 會處理路徑分隔符。
                            const staticOutputUrl = `http://127.0.0.1:5000/static_output/`;
                            
                            // 這裡我們直接使用後端傳來的 output_path_relative
                            // 因為後端現在也知道要返回相對於 output_folder 的路徑，
                            // 並且 send_file 會處理好斜線問題
                            const displayPath = staticOutputUrl + result.output_path_relative;

                            console.log("Inferred Media URL:", displayPath); // 調試輸出

                            let mediaElement;
                            if (result.is_video) { // 使用後端傳來的 is_video 判斷
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

    downloadOutputZipBtn.addEventListener('click', () => {
        window.location.href = 'http://127.0.0.1:5000/download_output_zip';
    });

    downloadLogZipBtn.addEventListener('click', () => {
        window.location.href = 'http://127.0.0.1:5000/download_log_zip';
    });

    cleanupBtn.addEventListener('click', async () => {
        displayStatus(cleanupStatus, '正在清除檔案...', 'info');
        try {
            const response = await fetch('http://127.0.0.1:5000/cleanup', {
                method: 'POST'
            });
            const data = await response.json();
            if (data.success) {
                displayStatus(cleanupStatus, data.message, 'success');
                uploadedFilesInfo = [];
                fileSelectionStatus.textContent = '請選擇檔案...';
                selectedPreviews.innerHTML = '';
                selectedPreviewsContainer.style.display = 'none';
                uploadedPreviews.innerHTML = '';
                inferenceResults.innerHTML = '';
                inferenceStatus.innerHTML = '';
                uploadStatus.innerHTML = '';
                downloadOutputZipBtn.disabled = true;
                downloadLogZipBtn.disabled = true;
                fileInput.value = '';
                inferBtn.disabled = true;
            } else {
                displayStatus(cleanupStatus, `清除失敗: ${data.message}`, 'error');
            }
        } catch (error) {
            displayStatus(cleanupStatus, `清除錯誤: ${error.message}`, 'error');
            console.error('Cleanup error:', error);
        }
    });

    displayStatus(fileSelectionStatus, '請選擇檔案...', 'info');
    selectedPreviewsContainer.style.display = 'none';
});