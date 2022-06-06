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
