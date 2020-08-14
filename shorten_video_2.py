from fire import Fire
from moviepy.editor import VideoFileClip

def main(video, output, start, end): #start and end in seconds
    clip = VideoFileClip(video).subclip(start, end)
    clip.write_videofile(output)

if __name__ == '__main__':
    Fire(main)
