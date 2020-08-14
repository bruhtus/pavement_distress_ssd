from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from fire import Fire

def main(video, output, start_timestamp, end_timestamp):
    ffmpeg_extract_subclip(video, start_timestamp, end_timestamp, targetname=output)
    #start_timestamp and end_timestamp is in second

if __name__ == '__main__':
    Fire(main)
