import os
import glob
import shutil
from fire import Fire
from tqdm import tqdm
from natsort import natsorted

def main(input_path1, input_path2, input_path3, save_path):
    all_frames = []

    for i in tqdm(range(3)):
        input_path = [input_path1, input_path2, input_path3]
        print(f'input_path{i+1}')
        for frame in tqdm(glob.glob(f'{input_path[i]}/*.jpg')):
            all_frames.append(frame)

    frame_sorted = natsorted(all_frames)

    for i in tqdm(range(len(frame_sorted))):
        filename = os.path.join(frame_sorted[i])
        dst_dir = os.path.join(save_path, f'frame_{i+1}.jpg')
        shutil.copy(filename, dst_dir)

if __name__ == '__main__':
    Fire(main)
