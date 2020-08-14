import cv2
import time
from fire import Fire
from tqdm import tqdm

def main(video_file, path_save, speed): # the lower the speed the fastest the frame_rates, speed = 0 (pause)
    vidcap = cv2.VideoCapture(video_file)
    current_frame = 0 
    speed_frame = speed

    while (vidcap.isOpened()): 
        success, frame = vidcap.read() # success = retrival value for frame 
        length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) 
        print(f'Current Frame: {current_frame}/{length}')
        current_frame += 1 

        if success == True:
            cv2.imshow('Video', frame)
            if cv2.waitKey(speed) & 0xFF == ord('s'): # press s to save the frame
                cv2.imwrite(f"{path_save}/frame_{current_frame}.jpg", frame)

            elif cv2.waitKey(speed) & 0xFF == ord('q'): # press q to quit
                break

            elif cv2.waitKey(speed) & 0xFF == ord('w'): # play/pause
                if speed != 0:
                    speed = 0
                elif speed == 0:
                    speed = speed_frame

        else:
            vidcap.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    Fire(main)
