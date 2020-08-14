import os
import cv2
import glob
import time
import math
import torch
import tempfile
import numpy as np
import streamlit as st

from fire import Fire
from tqdm import tqdm
from datetime import datetime
from streamlit import caching
from vizer.draw import draw_boxes
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

from ssd.config import cfg
from ssd.data.datasets import MyDataset
from ssd.utils.checkpoint import CheckPointer
from ssd.data.transforms import build_transforms
from ssd.modeling.detector import build_detection_model

def main():
    st.title('Pavement Distress Detector')
    st.markdown(get_file_content_as_string('./introduction.md'))
    st.sidebar.markdown(get_file_content_as_string('./documentation.md'))
    caching.clear_cache()
    video = video_uploader('./input')
    config = config_uploader('./configs')
    output_dir = checkpoint_folder('./outputs')

    filename = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{os.path.splitext(os.path.basename(config))[0]}"
    output_file = './results'
    score_threshold = st.slider('Confidence Threshold', 0.0, 1.0, 0.5)
    fps_threshold = st.slider('Counting Every (frames)', 10, 30, 20) 
    video_filename = f'{output_file}/{filename}.mp4'
    labels_filename = f'{output_file}/{filename}.txt'
    
    if st.button('Click here to run'):
        if (os.path.isdir(video) == False and os.path.isdir(config) == False and output_dir != './outputs/'):
            class_name = ('__background__',
                    'lubang', 'retak aligator', 'retak melintang', 'retak memanjang')

            cfg.merge_from_file(config)
            cfg.freeze()

            ckpt = None
            device = torch.device(cfg.MODEL.DEVICE)
            model = build_detection_model(cfg)
            model.to(device)

            checkpoint = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
            checkpoint.load(ckpt, use_latest=ckpt is None)
            weight_file = ckpt if ckpt else checkpoint.get_checkpoint_file()
            st.write(f'Loading weight from {weight_file}')
            cpu_device = torch.device('cpu')
            transforms = build_transforms(cfg, is_train=False)
            model.eval()

            clip = VideoFileClip(video)

            with tempfile.NamedTemporaryFile(suffix='.avi') as temp: #using temporary file because streamlit can't read opencv video result
                temp_name = temp.name
                pavement_distress(video, clip, fps_threshold, score_threshold, temp_name, labels_filename, transforms, model, device, cpu_device, class_name)

        elif (os.path.isdir(video) == True and os.path.isdir(config) == False and output_dir != './outputs/'):
            st.warning('Please select video file')

        elif (os.path.isdir(video) == True and os.path.isdir(config) == True and output_dir != './outputs/'):
            st.warning('Please select video file and config file')

        elif (os.path.isdir(video) == False and os.path.isdir(config) == True and output_dir != './outputs/'):
            st.warning('Please select config file')

        elif (os.path.isdir(video) == True and os.path.isdir(config) == False and output_dir == './outputs/'):
            st.warning('Please select video file and checkpoint folder')

        elif (os.path.isdir(video) == False and os.path.isdir(config) == False and output_dir == './outputs/'):
            st.warning('Please select checkpoint folder')

        elif (os.path.isdir(video) == False and os.path.isdir(config) == True and output_dir == './outputs/'):
            st.warning('Please select config file and checkpoint folder')

        else:
            st.warning('Please select video file, config file, and checkpoint folder')

def get_file_content_as_string(path):
    response = os.open(path, os.O_RDWR)
    string = os.read(response, os.path.getsize(response))
    return string.decode()

def video_uploader(folder_path):
    filenames = next(os.walk(folder_path))[2]
    #files = [f for f in filenames if f.endswith('.mp4')]
    #files.insert(0, '')
    filenames.insert(0, '')
    selected_files = st.selectbox('Select a video file', filenames)
    return os.path.join(folder_path, selected_files)

def config_uploader(folder_path):
    filenames = next(os.walk(folder_path))[2]
    files = [f for f in filenames if f.endswith('.yaml')]
    files.insert(0, '')
    selected_files = st.selectbox('Select a config file', files)
    return os.path.join(folder_path, selected_files)

def checkpoint_folder(folder_path):
    foldername = next(os.walk(folder_path))[1]
    foldername.insert(0, '')
    selected_folder = st.selectbox('Select a checkpoint folder', foldername)
    return os.path.join(folder_path, selected_folder)

def pavement_distress(video_name, cap, fps_threshold, threshold, output_video, output_labels, transforms, model, device, cpu_device, class_name):
    height, width = cap.h, cap.w
    #out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'MJPG'), 20.0, (width, height)) #DIVX for .MP4
    #f_output = open(output_labels, 'a+')

    all_labels = []
    fps = 1
    counting_fps = fps_threshold

    latest_iteration = st.empty()
    bar = st.progress(0)
    count = 0 #counting progress bar
    num_frames = cap.reader.nframes
    progress = 100/num_frames

    threshold_display = st.empty()
    threshold_display.text(f'Confidence Threshold: {threshold}')
    aligator = st.empty()
    memanjang = st.empty()
    melintang = st.empty()
    lubang = st.empty()
    total_labels = st.empty()

    aligator_count = 0
    memanjang_count = 0
    melintang_count = 0
    lubang_count = 0
    total_labels_count = 0

    for frame in cap.iter_frames():
        count = count + progress
        if count < 100:
            latest_iteration.text(f'Process: {math.ceil(count)}%')
            bar.progress(math.ceil(count))
        else:
            latest_iteration.text(f'Process: {math.floor(count)}%')
            bar.progress(math.floor(count))

        frames = transforms(frame)[0].unsqueeze(0)

        with torch.no_grad():
            result = model(frames.to(device))[0]
            result = result.resize((width, height)).to(cpu_device).numpy()
            boxes, labels, scores = result['boxes'], result['labels'], result['scores']

        indices = scores > threshold
        boxes = boxes[indices]
        labels = labels[indices]
        scores = scores[indices]

        if fps == fps_threshold:
            drawn_frame = draw_boxes(frame, boxes, labels, scores, class_name).astype(np.uint8)
            #cv2.imwrite(f'./results/{os.path.splitext(os.path.basename(video_name))[0]}/{counting_fps}_frames/frame_{fps}.jpg', drawn_frame)
            cv2.imwrite(f'./results/{os.path.splitext(os.path.basename(video_name))[0]}/confidence_threshold/{threshold}_threshold/frame_{fps}.jpg', drawn_frame)
            print(f'./results/{os.path.splitext(os.path.basename(video_name))[0]}/{threshold}_threshold/frame_{fps}.jpg')

            fps_threshold += counting_fps
            fps += 1

        else:
            fps += 1

if __name__ == '__main__':
    Fire(main)
