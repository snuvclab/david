import os
import json
import requests
import argparse
import time, datetime
import argparse
from glob import glob
import cv2
import numpy as np
from constants.config import CATEGORY2GENERATIONCONFIG
from constants.videos import headers, mode, SELECTED_IMAGE_PATHS
import base64

def gereate_videos(
    dataset,
    category,
    image_dir,
    video_save_dir,
    repeat,
    skip_done
):
    try: rgb_pths = SELECTED_IMAGE_PATHS[dataset][category]
    except: rgb_pths = []

    for rgb_pth in rgb_pths:
        for i in range(repeat):
            category = rgb_pth.split("/")[-3]
            save_dir = rgb_pth.replace(image_dir, video_save_dir).replace(".png", "")
            save_pth = f"{save_dir}/{i:05d}.mp4"

            prompt = CATEGORY2GENERATIONCONFIG[category]["prompt"]

            if skip_done and os.path.exists(save_pth): continue
            os.makedirs(save_dir, exist_ok=True)

            with open(rgb_pth, 'rb') as img:
                response = requests.post(
                    'https://api.imgur.com/3/upload',
                    headers=headers,
                    files={'image': img}
                )
                data = response.json()
                start_frame_url = data['data']['link']

                name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                os.makedirs(f'cache/results/{name}', exist_ok=True)

                os.system(f'sh scripts/videos/post_i2v.sh \"{prompt}\" \"{start_frame_url}\" > cache/results/{name}/post.json')

                # get video
                with open(f'cache/results/{name}/post.json', 'r') as f:
                    post = json.load(f)
                    assert post['code'] == 200, "wrong format" # success
                    task_id = post['data']['task_id']
                    print('Generating video...')


                # wait until generated
                time.sleep(300)
                while True:
                    os.system(f'sh scripts/videos/get.sh {task_id} > cache/results/{name}/get.json')
                    with open(f'cache/results/{name}/get.json', 'r') as f:
                        get = json.load(f)
                        if get['data']['status'] == 'completed':
                            print('Video generated!')
                            r = requests.get(get['data']['works'][0]['resource']['resourceWithoutWatermark'])
                            with open(save_pth, 'wb') as o:
                                o.write(r.content)
                            break
                        else:
                            time.sleep(30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## save directories
    parser.add_argument("--dataset", type=str, default="ComAsset")
    parser.add_argument("--category", type=str, default="barbell")
    parser.add_argument("--image_dir", type=str, default="results/generation/images")
    parser.add_argument("--video_save_dir", type=str, default="results/generation/videos_kling")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--skip_done", action="store_true")

    args = parser.parse_args()
    gereate_videos(
        dataset=args.dataset,
        category=args.category,
        image_dir=args.image_dir,
        video_save_dir=args.video_save_dir,
        repeat=args.repeat,
        skip_done=args.skip_done
    )