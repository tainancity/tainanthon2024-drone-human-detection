document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const uploadStatus = document.getElementById('uploadStatus');

    const fileSelectionStatus = document.getElementById('fileSelectionStatus');

    const uploadedFilesDisplay = document.getElementById('uploadedFilesDisplay');
    const uploadedPreviews = document.getElementById('uploadedPreviews');

    const inferSection = document.getElementById('inferSection');

    const inferBtn = document.getElementById('inferBtn');
    const inferenceStatus = document.getElementById('inferenceStatus');
    const interruptBtn = document.getElementById('interruptInferBtn');
    const inferenceResults = document.getElementById('inferenceResults');
    const downloadOutputZipBtn = document.getElementById('downloadOutputZipBtn');
    const downloadLogZipBtn = document.getElementById('downloadLogZipBtn');
    const cleanupBtn = document.getElementById('cleanupBtn');
    const cleanupStatus = document.getElementById('cleanupStatus');

    const inferenceProgressBarContainer = document.getElementById('inferenceProgressBarContainer');
    const inferenceProgressBar = document.getElementById('inferenceProgressBar');
    const inferenceProgressText = document.getElementById('inferenceProgressText');
    const currentFileInference = document.getElementById('currentFileInference');

    const liveInferenceImage = document.createElement('img');
    liveInferenceImage.style.maxWidth = '100%'; // limit width to container
    liveInferenceImage.style.height = 'auto';
    liveInferenceImage.style.display = 'none'; // not displayed initially
    liveInferenceImage.alt = 'Live Inference';
    inferenceProgressBarContainer.appendChild(liveInferenceImage);

    const easterEggTrigger = document.getElementById('easterEggTrigger');
    const easterEggDiv = document.getElementById('easterEgg');
    const youtubeIframe = easterEggDiv.querySelector('iframe');

    if (easterEggTrigger && easterEggDiv && youtubeIframe) {
        easterEggTrigger.addEventListener('click', () => {
            if (easterEggDiv.style.display === 'none') {
                easterEggDiv.style.display = 'block';
                youtubeIframe.src = "https://www.youtube.com/embed/dQ_d_VKrFgM?autoplay=1&controls=0";
            } else {
                easterEggDiv.style.display = 'none';
                youtubeIframe.src = "";
            }
        });
    }

    let uploadedFilesInfo = [];
    let socket;

    function initSocketIO() {
        socket = io('http://127.0.0.1:5000'); 

        socket.on('connect', () => {
            console.log('Connected to Socket.IO server!');
            displayStatus(inferenceStatus, '已連接到伺服器。', 'info');
        });

        socket.on('disconnect', () => {
            console.log('Disconnected from Socket.IO server.');
            displayStatus(inferenceStatus, '已斷開與伺服器的連接。', 'error');
        });

        socket.on('connect_error', (error) => {
            console.error('Socket.IO connection error:', error);
            displayStatus(inferenceStatus, `連接伺服器失敗: ${error.message}`, 'error');
        });

        // listen for inference progress updates
        socket.on('inference_progress', (data) => {
            console.log('Progress:', data.progress, 'Filename:', data.filename);
            const progress = data.progress;
            const filename = data.filename; //  UUID 
            const imageData = data.frame; // base64 encoded image data
            
            // find the original file name from uploadedFilesInfo
            const originalFile = uploadedFilesInfo.find(f => f.uuid_name === filename);
            const originalFileName = originalFile ? originalFile.original_name : filename;

            inferenceProgressBar.style.width = `${progress}%`;
            inferenceProgressText.textContent = `${progress}%`;
            currentFileInference.textContent = `正在推理: ${originalFileName} (${progress}%)`;
            inferenceProgressBarContainer.style.display = 'block'; // progress bar container should be visible

            if (progress === 100) {
                inferenceProgressText.style.color = '#333'; 
                liveInferenceImage.style.display = 'none';
            } else {
                inferenceProgressText.style.color = 'white';
                if (imageData) {
                    liveInferenceImage.src = `data:image/jpeg;base64,${imageData}`;
                    liveInferenceImage.style.display = 'block';
                } else {
                    liveInferenceImage.style.display = 'none';
                }
            }
        });

        // listen for batch inference started event
        socket.on('batch_inference_started', (data) => {
            displayStatus(inferenceStatus, `開始推理 ${data.total_files} 個檔案...`, 'info');
            // reset progress bar
            inferenceProgressBar.style.width = '0%';
            inferenceProgressText.textContent = '0%';
            currentFileInference.textContent = '';
            inferenceProgressBarContainer.style.display = 'block';
            liveInferenceImage.style.display = 'none';
            inferBtn.disabled = true;
            interruptBtn.disabled = false;
        });

        // listen for single file inference started event
        socket.on('file_inference_started', (data) => {
            displayStatus(inferenceStatus, `正在處理檔案 ${data.index + 1} / ${uploadedFilesInfo.length}: ${data.filename}...`, 'info');
            inferenceProgressBar.style.width = '0%';
            inferenceProgressText.textContent = '0%';
            currentFileInference.textContent = `正在推理: ${data.filename} (0%)`;
        });

        // listen for file inference completed event
        socket.on('file_inference_completed', (result) => {
            // console.log('File inference completed:', result);
            displayStatus(inferenceStatus, `檔案 ${result.original_name} 推理完成！`, 'success');
            // add result to the display
            addInferenceResultToDisplay(result);
        });

        // listen for batch inference completed event
        socket.on('batch_inference_completed', (data) => {
            displayStatus(inferenceStatus, '所有檔案推理完成！', 'success');
            inferenceProgressBarContainer.style.display = 'none';
            liveInferenceImage.style.display = 'none';
            downloadOutputZipBtn.disabled = false;
            downloadLogZipBtn.disabled = false;
            inferBtn.disabled = false;
            interruptBtn.disabled = true;
            
        });

        // listen for inference error event
        socket.on('inference_error', (error) => {
            console.error('Inference error from server:', error);
            displayStatus(inferenceStatus, `推理過程中發生錯誤: ${error.message}`, 'error');
            inferenceProgressBarContainer.style.display = 'none';
            liveInferenceImage.style.display = 'none';
            inferBtn.disabled = false;
            interruptBtn.disabled = true;
        });

        // listen for inference interrupted event
        socket.on('inference_interrupted', (data) => {
            console.log('Inference interrupted:', data);
            displayStatus(inferenceStatus, `推理已中斷: ${data.message}`, 'info');
            inferenceProgressBarContainer.style.display = 'none';
            liveInferenceImage.style.display = 'none';
            inferBtn.disabled = false;
            interruptBtn.disabled = true;
        });
    }

    // display inference result in the results section
    function addInferenceResultToDisplay(result) {
        const itemDiv = document.createElement('div');
        itemDiv.className = 'inference-item';
        itemDiv.innerHTML = `<h3>${result.original_name}</h3>`;

        if (result.success) {
            itemDiv.innerHTML += `<p style="color: green;">${result.message}</p>`;
            if (result.output_path_relative) {
                const staticOutputUrl = `http://127.0.0.1:5000/static_output/`;
                const displayPath = staticOutputUrl + result.output_path_relative;

                console.log("Inferred Media URL:", displayPath);

                let mediaElement;
                if (result.is_video) {
                    mediaElement = document.createElement('video');
                    mediaElement.controls = true;
                    mediaElement.autoplay = false;
                    mediaElement.loop = false;
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
    }

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
        fileSelectionStatus.textContent = '';

        uploadedFilesInfo = [];
        inferBtn.disabled = true;

        const files = event.target.files;
        if (files.length === 0) {
            displayStatus(fileSelectionStatus, '請選擇檔案...', 'info');
            return;
        }

        let fileNames = Array.from(files).map(file => file.name).join(', ');
        displayStatus(fileSelectionStatus, `已選擇檔案: ${fileNames}`, 'info');

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
                uploadedFilesDisplay.style.display = 'block';
                inferSection.style.display = 'block';

                uploadedFilesInfo.forEach(file => {
                    const itemDiv = document.createElement('div');
                    itemDiv.className = 'media-preview-item';
                    itemDiv.innerHTML = `<h4>${file.original_name}</h4>`;

                    if (file.success) {
                        const staticInputUrl = `http://127.0.0.1:5000/static_input/`;
                        // assume file.uuid_name include file extension
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
        if (uploadedFilesInfo.length === 0) {
            displayStatus(inferenceStatus, '請先上傳檔案。', 'error');
            return;
        }
        // clean up previous results
        inferenceResults.innerHTML = '';
        inferBtn.disabled = true;
        interruptBtn.disabled = false;
        downloadOutputZipBtn.disabled = true;
        downloadLogZipBtn.disabled = true;
        
        // send start inference event
        socket.emit('start_inference_batch');
    });

     interruptBtn.addEventListener('click', () => {
        displayStatus(inferenceStatus, '正在發送中斷請求...', 'info');
        interruptBtn.disabled = true;
        socket.emit('interrupt_inference');
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
                uploadedPreviews.innerHTML = '';
                inferenceResults.innerHTML = '';
                inferenceStatus.innerHTML = '';
                uploadStatus.innerHTML = '';
                uploadedFilesDisplay.style.display = 'none';
                inferSection.style.display = 'none';
                liveInferenceImage.style.display = 'none';
                downloadOutputZipBtn.disabled = true;
                downloadLogZipBtn.disabled = true;
                fileInput.value = '';
                inferBtn.disabled = true;              
                inferenceProgressBarContainer.style.display = 'none';
            } else {
                displayStatus(cleanupStatus, `清除失敗: ${data.message}`, 'error');
            }
        } catch (error) {
            displayStatus(cleanupStatus, `清除錯誤: ${error.message}`, 'error');
            console.error('Cleanup error:', error);
        }
    });

    interruptBtn.disabled = true; // deisable interrupt button initially
    displayStatus(fileSelectionStatus, '請選擇檔案...', 'info');

    initSocketIO();
});