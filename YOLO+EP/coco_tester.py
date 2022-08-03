# IMPORT
import os
import numpy as np
import utils

from itertools import groupby
from coco_files.coco import COCO
from PIL import Image

import warnings
warnings.simplefilter("ignore", FutureWarning)

# FUNZIONI
def model_inference(abs_ds_path, model):
    import subprocess
    import re
    import sys

    list_of_file = []
    # popolo list_of_file prendendo solo i file jpg
    for file in os.listdir(abs_ds_path):
        if os.path.isfile(os.path.join(abs_ds_path, file)):
            list_of_file.append(f'{abs_ds_path}\\{file}')
    
    # se il processo non si trova già in EFFICIENTPOSE_PATH viene fatto il cambio di directory
    try:
        os.chdir(EFFICIENTPOSE_PATH)
    except:
        pass

    times = []
    # per ogni file viene creata una immagine contenente la stima della posa
    for file in list_of_file:
        # recuperare l'output del sottoprocesso come stringa
        stdout = subprocess.check_output(f'python {EFFICIENTPOSE_MAIN} --model={model} --path="{file}" --framework={FRAMEWORK}', shell=True).decode(sys.stdout.encoding)
        # ricerco questo pattern dentro la stringa
        res = re.search(r'Image processed in [-+]?(?:\d*\.\d+|\d+) seconds', stdout).group(0)
        
        print(f'\n{res}\n')

        # estraggo il numero float e lo aggiungo a times
        times.append(float(re.findall(r'[-+]?(?:\d*\.\d+|\d+)', res)[0]))
    
    print(f"\nTempo medio di analisi è {sum(times) / len(times)}")

def move_inference_in_model_dir(abs_ds_path, model):
    # errore se la dir esiste già
    try:
        os.mkdir(f'{abs_ds_path}\\{model}')
    except:
        pass
    
    for file in os.listdir(abs_ds_path):
        if 'tracked' in file:
            os.replace(f'{abs_ds_path}\\{file}', f'{abs_ds_path}\\{model}\\{file}')

def search_person_image(annotations, n_of_img, num_of_people):
    '''
    A partire dalle annotazioni determina se nell'immagine è presente più di una persona
    '''
    analyzed = {}
    removed  = set()
    for obj in annotations:
        if obj['num_keypoints'] < 10:
            continue
    
        analyzed[obj['image_id']] = analyzed.get(obj['image_id'], 0) + 1
        
    # cerco le immagini che hanno un certo numero di persone
    tmp = []
    for k in analyzed:
        if analyzed[k] == num_of_people:
            tmp.append(k)
    
    analyzed = set(tmp)
    return list(analyzed - removed)[0:n_of_img]

# COSTANTI
ANNOTATIONS_PATH = './coco_files/annotations/person_keypoints_val2017.json'
DS_PATH = './coco_files/dataset/'
EFFICIENTPOSE_PATH = './'
EFFICIENTPOSE_MAIN = 'track.py'
MODELs = ['RT','I','II', 'III', 'IV']

# VARIABILI
FRAMEWORK = 'tflite' #, 'keras' #"torch"# "tf", #
MODEL = MODELs[0]
NUMBER_OF_IMAGES = 2
TAU_FOR_PRETTIFY = 0.5

NUM_OF_PEOPLE = 2

# True lavora su un solo modello (MODEL), False crea un plot che compara i diversi modelli
ANALYZE_ONE_MODEL = True

# MAIN
if __name__ == "__main__":
    coco=COCO(ANNOTATIONS_PATH)
    
    # recupero l'id della categoria persona
    person_cat_id = coco.getCatIds(catNms=['person'])[0]
    
    # recupero l'id delle immagini di categoria persona
    img_ids = coco.getImgIds(catIds=[person_cat_id])
    
    # carico le annotazioni delle immagini con id img_ids
    annotations = coco.loadAnns(ids=coco.getAnnIds(imgIds=img_ids))
    
    # search_person_image cerca le immagini che hanno un certo numero di persone al suo interno
    img_ids = search_person_image(annotations, NUMBER_OF_IMAGES, NUM_OF_PEOPLE)
    
    # carico le annotazioni di interesse cioè quelle che rispettano i requisiti di search_person_image
    annotations = coco.loadAnns(ids=coco.getAnnIds(imgIds=img_ids))

    # path assoluto della cartella del dataset
    abs_ds_path = os.path.abspath(DS_PATH)

    # scarica le immagini se non sono già presenti nel dataset
    coco.download(imgIds=img_ids, tarDir=abs_ds_path)
    
    # vengono create le sole immagini generate dal modello
    if ANALYZE_ONE_MODEL:
        model_inference(abs_ds_path, MODEL) 
        move_inference_in_model_dir(abs_ds_path, MODEL)
    else:
        for m in MODELs:
            model_inference(abs_ds_path, m)
            move_inference_in_model_dir(abs_ds_path, m)
            