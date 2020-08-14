from fire import Fire
from natsort import natsorted
from tqdm import tqdm
import cv2
import os
import glob

def main(input_path1, input_path2, input_path3, save_path, target_size):
    all_frames = []

    for i in tqdm(range(3)):
        input_path = [input_path1, input_path2, input_path3]
        print(f'input_path{i+1}')
        for frame in tqdm(glob.glob(f'{input_path[i]}/*.jpg')):
            all_frames.append(frame)

    frame_sorted = natsorted(all_frames)

    resize(frame_sorted, save_path, target_size)

def resize(input_path, save_path, target_size):
    for frames in tqdm(range(len(input_path))):
        filename = os.path.join(input_path[frames])
        frame = cv2.imread(filename)
        frame_small = cv2.resize(frame, target_size)
        #frame_name = os.path.basename(filename)
        cv2.imwrite(f'{save_path}/frame_{frames+1}.jpg', frame_small)

if __name__ == '__main__':
    Fire(main)
