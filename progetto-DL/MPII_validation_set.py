import json
import os, shutil

""" f = open("./MPII/valid.json")
val_anno = json.load(f)

images = []
for i in val_anno:
    images.append(i["image"]) """

f = open("./MPII/sub_valid.txt")
images = eval(f.read())

print(len(images))

DS_PATH = './MPII'
abs_ds_path = os.path.abspath(DS_PATH)

#image_folder = f'{abs_ds_path}/valid/'
image_folder = f'{abs_ds_path}/sub_valid/'
for img in images:
    src = f'{abs_ds_path}/images/{img}'
    shutil.copy(src, image_folder+img)

saved = os.listdir(image_folder)
print(len(saved))
print(len(set(images)))

f.close()


""" import cv2

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
heights = [cv2.imread(os.path.join(image_folder, image)).shape[0] for image in images]
widths = [cv2.imread(os.path.join(image_folder, image)).shape[1] for image in images]
height = max(heights)
width = max(widths)
print(height, width) """

# cat *.jpg | ffmpeg -f image2pipe -i - valid.mp4

# frame= 2729 fps= 12 q=-1.0 Lsize=  178311kB time=00:01:49.08 bitrate=13391.3kbits/s
# video:178297kB audio:0kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 0.007840%

# frame= 1199 fps= 23 q=-1.0 Lsize=   46471kB time=00:00:47.88 bitrate=7950.9kbits/s
# video:46465kB audio:0kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 0.013333%
