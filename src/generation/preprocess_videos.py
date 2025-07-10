from moviepy.editor import VideoFileClip
from glob import glob
import os
import argparse
from constants.config import WIDTH, HEIGHT

def resize_and_save(input_file, output_file, new_width, new_height):
    clip = VideoFileClip(input_file)
    resized_clip = clip.resize(newsize=(new_width, new_height))
    resized_clip.write_videofile(output_file, codec='libx264', preset='slow', bitrate='5000k', audio=True)
    clip.close()
    resized_clip.close()


def resize_videos(
    dataset,
    category,
    video_dir,
    video_save_dir,
    skip_done
):
    video_pths = glob(f"{video_dir}/{dataset}/{category}/*/*.mp4")
    for input_file in video_pths:
        output_file = input_file.replace(video_dir, video_save_dir)
        if skip_done and os.path.exists(output_file): continue

        video_resize_save_dir = "/".join(output_file.split("/")[:-1])
        os.makedirs(video_resize_save_dir, exist_ok=True)

        resize_and_save(input_file, output_file, WIDTH, HEIGHT)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="ComAsset")
    parser.add_argument("--category", type=str, default="barbell")
    parser.add_argument("--video_dir", type=str, default="results/generation/videos_kling")
    parser.add_argument("--video_save_dir", type=str, default="results/generation/videos")
    parser.add_argument("--skip_done", action="store_true")

    args = parser.parse_args()
    resize_videos(
        dataset=args.dataset,
        category=args.category,
        video_dir=args.video_dir,
        video_save_dir=args.video_save_dir,
        skip_done=args.skip_done
    )