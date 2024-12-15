import torch
import torch.nn as nn 
import torchvision.transforms as T
from torch.cuda.amp import autocast
import numpy as np 
from PIL import Image, ImageDraw, ImageFont
import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS
import numpy as np
import cv2
from tqdm import tqdm
import time
import argparse
def postprocess(labels, boxes, scores, iou_threshold=0.55):
    def calculate_iou(box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area if union_area != 0 else 0
        return iou
    merged_labels = []
    merged_boxes = []
    merged_scores = []
    used_indices = set()
    for i in range(len(boxes)):
        if i in used_indices:
            continue
        current_box = boxes[i]
        current_label = labels[i]
        current_score = scores[i]
        boxes_to_merge = [current_box]
        scores_to_merge = [current_score]
        used_indices.add(i)
        for j in range(i + 1, len(boxes)):
            if j in used_indices:
                continue
            if labels[j] != current_label:
                continue  
            other_box = boxes[j]
            iou = calculate_iou(current_box, other_box)
            if iou >= iou_threshold:
                boxes_to_merge.append(other_box.tolist())  
                scores_to_merge.append(scores[j])
                used_indices.add(j)
        xs = np.concatenate([[box[0], box[2]] for box in boxes_to_merge])
        ys = np.concatenate([[box[1], box[3]] for box in boxes_to_merge])
        merged_box = [np.min(xs), np.min(ys), np.max(xs), np.max(ys)]
        merged_score = max(scores_to_merge)
        merged_boxes.append(merged_box)
        merged_labels.append(current_label)
        merged_scores.append(merged_score)
    return [np.array(merged_labels)], [np.array(merged_boxes)], [np.array(merged_scores)]
def slice_image(image, slice_height, slice_width, overlap_ratio):
    img_width, img_height = image.size
    
    slices = []
    coordinates = []
    step_x = int(slice_width * (1 - overlap_ratio))
    step_y = int(slice_height * (1 - overlap_ratio))
    
    for y in range(0, img_height, step_y):
        for x in range(0, img_width, step_x):
            box = (x, y, min(x + slice_width, img_width), min(y + slice_height, img_height))
            slice_img = image.crop(box)
            slices.append(slice_img)
            coordinates.append((x, y))
    return slices, coordinates
def merge_predictions(predictions, slice_coordinates, orig_image_size, slice_width, slice_height, threshold=0.30):
    merged_labels = []
    merged_boxes = []
    merged_scores = []
    orig_height, orig_width = orig_image_size
    for i, (label, boxes, scores) in enumerate(predictions):
        x_shift, y_shift = slice_coordinates[i]
        scores = np.array(scores).reshape(-1)
        valid_indices = scores > threshold
        valid_labels = np.array(label).reshape(-1)[valid_indices]
        valid_boxes = np.array(boxes).reshape(-1, 4)[valid_indices]
        valid_scores = scores[valid_indices]
        for j, box in enumerate(valid_boxes):
            box[0] = np.clip(box[0] + x_shift, 0, orig_width)  
            box[1] = np.clip(box[1] + y_shift, 0, orig_height)
            box[2] = np.clip(box[2] + x_shift, 0, orig_width)  
            box[3] = np.clip(box[3] + y_shift, 0, orig_height) 
            valid_boxes[j] = box
        merged_labels.extend(valid_labels)
        merged_boxes.extend(valid_boxes)
        merged_scores.extend(valid_scores)
    return np.array(merged_labels), np.array(merged_boxes), np.array(merged_scores)
def draw(images, labels, boxes, scores, thrh = 0.6, path = ""):
    for i, im in enumerate(images):
        box_count = 0
        draw = ImageDraw.Draw(im)
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scores[i][scr > thrh]
        box_count += len(box)
        for j,b in enumerate(box):
            draw.rectangle(list(b), outline='red',width=5)
            draw.text((b[0], b[1]), text=f"human: {lab[j].item()} {round(scrs[j].item(),2)}", font=ImageFont.load_default(), fill='blue')
        return im, box_count
def initModel(args):
    print("Load parameters")
    cfg = YAMLConfig(args.config, resume=args.resume)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')
    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)
    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
    
    print("Initialize model")
    model = Model().to(args.device)
    return model
            
def Inference(args):
    """main
    """
    print("Load parameters")
    cfg = YAMLConfig(args.config, resume=args.resume)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')
    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)
    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
    
    print("Initialize model")
    model = Model().to(args.device)
    # Read the video and verify the code is correct
    if args.video:
        cap = cv2.VideoCapture(args.imfile)
        
        # get the fps, w, h, if the input video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # set output video type .mp4
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video = os.path.split(args.imfile)[-1]
        video_name = video.split('.')[0]

        os.makedirs(os.path.join(args.outputdir,video_name), exist_ok=True)
        new_path = os.path.join(args.outputdir,video_name)

        output_video = cv2.VideoWriter(os.path.join(new_path,"output.mp4"), fourcc, fps, (width, height))
        if not cap.isOpened():
            print("cap can not open")
            exit()

        while cap.isOpened():
            for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), desc="Processing frames", ncols=100):
                ret, frame = cap.read()
                if not ret:
                    print("Frame end or can not read frame")
                    exit()
                    
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
                    
                detect_frame = draw([im_pil], labels, boxes, scores, 0.35)
                frame_out = cv2.cvtColor(np.array(detect_frame), cv2.COLOR_RGB2BGR)
                output_video.write(frame_out)

                # Press q to end the code
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                
            # cload all the windows
            cap.release()
            output_video.release()
            cv2.destroyAllWindows()
    else:
        # for root, dirs, files in os.walk(args.imfile):
        #     for image in tqdm(files):
        image = args.imfile
        img = cv2.imread(os.path.join(args.imfile,image))
        video_name = image.split('.')[0]
        
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
        detect_frame = draw([im_pil], labels, boxes, scores, 0.35)
        frame_out = cv2.cvtColor(np.array(detect_frame), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(args.outputdir,f"{video_name}.jpg"),frame_out)
        
def InitArgs(imfile, video, outputdir, device):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=r"rtdetr\configs\rtdetr\drone_sealand.yml")
    parser.add_argument('-r', '--resume', type=str, default=r"rtdetr\weights\checkpoint0029.pth")  #要改model暫時先從這裡改
    parser.add_argument('-f', '--imfile', type=str, default=imfile)
    parser.add_argument('-s', '--sliced', type=bool, default=False)
    parser.add_argument('-d', '--device', type=str, default=device)
    parser.add_argument('-nc', '--numberofboxes', type=int, default=25)
    parser.add_argument('-o', '--outputdir', type=str, default= outputdir)
    parser.add_argument('-v', '--video', type=bool, default=video)
    args = parser.parse_args()
    return args
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=r"rtdetr\configs\rtdetr\drone_sealand.yml")
    parser.add_argument('-r', '--resume', type=str, default=r"rtdetr\weights\checkpoint0022.pth")
    parser.add_argument('-f', '--imfile', type=str, default=r"testdata\703134876.184583.mp4")
    parser.add_argument('-s', '--sliced', type=bool, default=False)
    parser.add_argument('-d', '--device', type=str, default='cpu')
    parser.add_argument('-nc', '--numberofboxes', type=int, default=25)
    parser.add_argument('-o', '--outputdir', type=str, default= r"output")
    parser.add_argument('-v', '--video', type=bool, default=True)
    args = parser.parse_args()
    Inference(args)
