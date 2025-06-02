# VRSAR: A Real-Time, Offline Drone-Based Visual Recognition System for Search and Rescue

We present a Real-Time, Offline Drone-Based Visual Recognition system for Search And Rescue (VRSAR), a drone-based visual recognition system engineered for real-world SAR deployment.
## Functionality

- Upload videos or images
- Using RT-DETR v2 model
- Streamlit preview and render results
- Output videos and annotations download.

## Installation 

1. Clone the repo:

    ```bash
    git clone https://github.com/chengruchou/VRSAR
    ```

2. cd into the repo dir

    ```bash
    cd VRSAR
    ```

3. Create and activate conda env(recommended)

    ```bash
    conda create -n yourEnvName python=3.8  # higher than 3.8
    conda activate yourEnvName  
    ```

5. Install required packages

    **Windows:**
    ```bash
    install.bat
    ```

6. Setup pretrained weight
    Get compatible model weight file, and edit path in rtdetr/tools/infer.py 

## Usage

1.  Run the streamlit application 
    ```bash
    streamlit run backend/main.py --server.maxUploadSize 10000
    ```
2. Open the application in your browser `http://localhost:8501`(if it doesn't open automatically)
3. Follow onscreen instructions to upload files or activate real-time streaming.

## File structure 

- `main.py`: The main file
- `requirements.txt`: The required python libraries.
- `weights/`: Model weights(checkpoints)
- `outputFile/`: Inference results.

## Dependencies 

- Python 3.8+
- Streamlit
- ultralytics
- cv2
- CUDA Toolkit

## Licensing 

This project is licensed under the MIT license. Please read the [LICENSE](LICENSE) documents.
