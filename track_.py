from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
import json

import cv2
import torch
import numpy as np
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
# parser.add_argument('--video_name', default='', type=str,
#                     help='videos or image files')
args = parser.parse_args()


def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def main():

    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args.snapshot,
        map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    #=============== modified =================#
    # Load the JSON files from the 'masks' directory
    masks_dir = '/mnt/data-hdd/jieming/surgTool_dataset_part/masks'
    json_files = glob(os.path.join(masks_dir, '*.json'))

    for json_file in json_files:
        # Load the JSON data
        with open(json_file, 'r') as f:
            json_data = json.load(f)

        # Extract the video filename from the JSON file and replace the extension with '.mp4'
        video_filename = os.path.splitext(json_data["info"]["name"])[0] + '.mp4'

        # Get the path to the corresponding video
        video_path = f'/mnt/data-hdd/wa/dataset/SurgToolLoc_2022/surgtoolloc2022_dataset/_release/training_data/video_clips/{video_filename}'

        output_dir = os.path.join('/mnt/data-hdd/jieming/pysot/my_output', os.path.splitext(video_filename)[0])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)


        group_bbox_data=[]

        for group_data in json_data["objects"]:
            # Extract the bounding box information for the current group
            bbox = group_data["bbox"]
            init_rect = (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])

            print("====NEW OBJECT DETECTION====")
            # Create a subdirectory for each group
            group_dir = os.path.join(output_dir, group_data["group"])
            if not os.path.exists(group_dir):
                os.makedirs(group_dir)

            cv2.namedWindow(video_filename, cv2.WND_PROP_FULLSCREEN)
            frame_count = 0
            save_count = 0

            # reset first frame
            first_frame = True
            
            
            for frame in get_frames(video_path):
                if first_frame:
                    #===========modified==========#
                    print("==##==RECOGNIZED FIRST FRAME==##==")

                    # Use the provided bounding box for the first frame
                    tracker.init(frame, init_rect)
                    first_frame = False

                    # manual annotation 
                    # try:
                    #     init_rect = cv2.selectROI(video_name, frame, False, False)
                    # except:
                    #     exit()
                    # tracker.init(frame, init_rect)
                    # first_frame = False

                else:
                    outputs = tracker.track(frame)
                    if 'polygon' in outputs:
                        polygon = np.array(outputs['polygon']).astype(np.int32)
                        cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                                    True, (0, 255, 0), 3)
                        mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                        mask = mask.astype(np.uint8)
                        mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                        frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
                    else:
                        bbox = list(map(int, outputs['bbox']))
                        cv2.rectangle(frame, (bbox[0], bbox[1]),
                                    (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                                    (0, 255, 0), 3)
                    cv2.imshow(video_filename, frame)
                    cv2.waitKey(40)
                
                # Save the frame for every 24 frames and the corresponding bounding box output JSON file
                if frame_count % 24 == 0:
                    bbox_data = {
                            "class id": group_data["group"],
                            "bbox": bbox,
                        }
                    group_bbox_data.append(bbox_data)
                    # Save bounding box visualization image to the group subdirectory
                    bbox_image_path = os.path.join(group_dir, f'{video_filename.replace(".mp4","")}_{save_count:04d}.jpg')
                    cv2.imwrite(bbox_image_path, frame)

                    # Save the bounding box information for each saved frame as a separate JSON file
                    group_json_path = os.path.join(group_dir, f'{video_filename.replace(".mp4","")}_bbox_{save_count:04d}.json')
                    with open(group_json_path, 'w') as f:
                        json.dump(group_data, f, indent=4)

                    save_count += 1

                frame_count += 1
   
        


if __name__ == '__main__':
    main()
