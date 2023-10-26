import os
import random
import cv2
import numpy as np

data_dir = '/mnt/data-hdd/jieming/pysot/my_output'
txt_label_dir = '/mnt/data-hdd/jieming/new_txt_label'
num_clips = 15

if not os.path.exists(txt_label_dir):
    os.makedirs(txt_label_dir)
    
clip_dirs = os.listdir(data_dir)
random.shuffle(clip_dirs) 
clip_dirs = clip_dirs[:num_clips]

for clip_dir in clip_dirs:

  clip_path = os.path.join(data_dir, clip_dir)

  for root, dirs, files in os.walk(clip_path):
    
    clip_name = root.split('/')[-1]
    
    for class_dir in dirs:

        class_id = class_dir

        img_dir = os.path.join(root, class_dir)

        for img_file in os.listdir(img_dir):

            img_path = os.path.join(img_dir, img_file)

            if img_file.endswith('.jpg'):

                img = cv2.imread(img_path)

                # 搜索绿色像素
                lower = np.array([0,255,0], dtype=np.uint8)  
                upper = np.array([0,255,0], dtype=np.uint8)

                # 阈值处理  
                mask = cv2.inRange(img, lower, upper)

                # 降噪处理
                mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY |cv2.THRESH_OTSU)[1] 

                # 查找轮廓
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

                # Draw bounding boxes for the contours 
                for c in contours:
                    bbox = cv2.boundingRect(c)
                    x,y,w,h = bbox


                x_min = x
                y_min = y
                x_max = x + w
                y_max = y + h

                txt_filename = img_path.split('/')[-1].replace('.jpg','.txt')
                txt_path = os.path.join(txt_label_dir, txt_filename)

                with open(txt_path, 'a') as f:
                    f.write(str(class_id)+' '+str(x_min)+' '+str(y_min)+' '+str(x_max)+' '+str(y_max)+'\n')




