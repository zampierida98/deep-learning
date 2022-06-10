import json
import os, shutil

f = open("./MPII/valid.json")
val_anno = json.load(f)

images = []
for i in val_anno:
    images.append(i["image"])

print(len(images))

DS_PATH = './MPII'
abs_ds_path = os.path.abspath(DS_PATH)

""" for img in images:
    src = f'{abs_ds_path}/images/{img}'
    dst = f'{abs_ds_path}/valid/{img}'
    #print(src, dst)
    shutil.copy(src, dst) """

saved = os.listdir(f'{abs_ds_path}/valid/')
print(len(saved))
print(len(set(images)))

f.close()


image_folder = f'{abs_ds_path}/valid/'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
#heights = [cv2.imread(os.path.join(image_folder, image)).shape[0] for image in images]
#widths = [cv2.imread(os.path.join(image_folder, image)).shape[1] for image in images]
height = 1080#max(heights)
width = 1920#max(widths)
#print(height, width)

# cat *.jpg | ffmpeg -f image2pipe -i - valid.mp4

# frame= 2729 fps= 12 q=-1.0 Lsize=  178311kB time=00:01:49.08 bitrate=13391.3kbits/s
# video:178297kB audio:0kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 0.007840%
