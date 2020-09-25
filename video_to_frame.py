import cv2
from fire import Fire
from tqdm import tqdm

def main(video_file, path_to_save):
    cap = cv2.VideoCapture(video_file)
    property_id = cv2.CAP_PROP_FRAME_COUNT
    length = int(cv2.VideoCapture.get(cap, property_id))
    success, image = cap.read()
    for i in tqdm(range(length)):
        cv2.imwrite(f"{path_to_save}/frame_{i+1}.jpg", image)
        success, image = cap.read()

if __name__ == "__main__":
    Fire(main)
