# IMPORT
import os
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import utils

from itertools import groupby
from coco import COCO
from PIL import Image

import warnings
warnings.simplefilter("ignore", FutureWarning)

# FUNZIONI
def get_rows_from_csv(list_of_file, abs_ds_path, model):
    res = {}
    for file in list_of_file:
        with open(f'{abs_ds_path}\\{model}\\{file.split(".")[0]}_coordinates.csv', 'r') as fin:
            header = fin.readline().strip()
            header = header.replace("_x", ":x")
            header = header.replace("_y", ":y")
            fin.readline() # riga vuota
            value = fin.readline().strip()
            body_parts = list(zip(header.split(","), value.split(",")))
            body_parts.sort(key=lambda x: x[0])
            parts = {}
            for p in [list(i) for j, i in groupby(body_parts, lambda x: x[0].split(':')[0])]:
                # p = [[parte_corpo_x, val_x], [parte_corpo_y, val_y]]
                parts[p[0][0].split(":")[0]] = (p[0][1], p[1][1])
                res[file] = parts
    return res

def get_real_body_parts(inference, abs_ds_path):
    for img in inference:
        image = Image.open(f'{abs_ds_path}\\{img}')
        width = image.width
        height = image.height

        for bp in inference[img]:
            new_x = float(inference[img][bp][0]) * width
            new_y = float(inference[img][bp][1]) * height
            inference[img][bp] = (round(new_x), round(new_y))

    return inference

def create_csv_model_inference(abs_ds_path, model):
    list_of_file = []
    # popolo list_of_file prendendo solo i file jpg
    for file in os.listdir(abs_ds_path):
        if not file.endswith(".csv"):
            list_of_file.append(f'{abs_ds_path}\\{file}')
    
    # se il processo non si trova già in EFFICIENTPOSE_PATH viene fatto il cambio di directory
    try:
        os.chdir(EFFICIENTPOSE_PATH)
    except:
        pass

    # per ogni file viene creato il csv
    for file in list_of_file:
        os.system(f'python {EFFICIENTPOSE_MAIN} --model={model} --path="{file}" --store --framework={FRAMEWORK}')

def get_inference(list_of_img, abs_ds_path, model):
    # alcune img non hanno il campo annopoints per cui tali img non verranno recuperate da inference
    inference = get_rows_from_csv(list_of_img, abs_ds_path, model)
    return get_real_body_parts(inference, abs_ds_path)

def compare_models(annotations, abs_ds_path, metric_name, map_id_filename):
    metric = {'pck': utils.pck, 'pcp':utils.pcp, 'pdj':utils.pdj}
    _min,_max,step = 0, 1, 0.01
    res = {}
    ground_truth = get_rows_from_annotations(annotations, map_id_filename)
    for m in MODELs:
        print("Lavoro su EfficientPose", m, "...")

        inference = get_inference(ground_truth.keys(), abs_ds_path, m)
        X = [i for i in np.arange(_min, _max, step)]
        Y = [metric[metric_name](ground_truth, inference, x) for x in X]
        res[m] = (X,Y)

    utils.plot(res, metric_name)
    return res

def move_csv_in_model_dir(abs_ds_path, model):
    # errore se la dir esiste già
    try:
        os.mkdir(f'{abs_ds_path}\\{model}')
    except:
        pass
    
    for file in os.listdir(abs_ds_path):
        if file.endswith(".csv"):
            os.replace(f'{abs_ds_path}\\{file}', f'{abs_ds_path}\\{model}\\{file}')

def get_rows_from_annotations(annotations, map_id_filename):
    body_part_ids = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder",
                    "left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee",
                    "right_knee","left_ankle","right_ankle"]

    res = {}
    for obj in annotations:
        parts = {}
        # si lavora da 15 in quanto è corrispondente ai valori di left_shoulder
        for i in range(15,len(obj['keypoints']),3):
            parts[body_part_ids[i // 3]] = (round(obj['keypoints'][i]), round(obj['keypoints'][i+1]))        
        
        # bacino è collocato a metà tra le due anche
        parts['pelvis'] = ( round( (parts['left_hip'][0] + parts['right_hip'][0]) // 2),
                            round( (parts['left_hip'][1] + parts['right_hip'][1]) // 2)
                            )
        
        # il torace è un po' più in basso del valor medio tra i due 
        parts['thorax'] = ( round( (parts['left_shoulder'][0] + parts['right_shoulder'][0]) // 2),
                            round( (parts['left_shoulder'][1] + parts['right_shoulder'][1]) // 2)
                            )
        
        # distanza sulla x tra i due occhi è circa uguale alla distanza centro_tra_i_due_occhi e la parte_superiore_testa
        
        parts['head_top'] = (round( (obj['keypoints'][3] + obj['keypoints'][6]) // 2),
                            round( (obj['keypoints'][4] + obj['keypoints'][7]) // 2)
                            )
        distanza_occhi = abs(round(obj['keypoints'][6] - obj['keypoints'][3]))
        
        parts['head_top'] = (parts['head_top'][0] - distanza_occhi, parts['head_top'][1])        

        # upper_neck è collocato a metà tra naso e torace
        '''
        parts['upper_neck'] = ( round( (obj['keypoints'][0] + parts['thorax'][0]) // 2),
                            round( (obj['keypoints'][1] + parts['thorax'][1]) // 2)
                            )'''
        
        res[map_id_filename[ obj['image_id'] ] ] = parts

    return res

def search_person_image(annotations, n_of_img, single_person=True):
    '''
    A partire dalle annotazioni determina se nell'immagine è presente più di una persona
    '''
    analyzed = set()
    removed  = set()
    for obj in annotations:
        if obj['num_keypoints'] != 17:
            removed.add(obj['image_id'])
            continue
        if obj['image_id'] in analyzed:
            # se voglio un multi-person allora non aggiungo l'id dell'img nel caso in cui
            # ci fosse l'annotazione di più persone per la stessa img.
            if single_person:
                removed.add(obj['image_id'])
        else:    
            analyzed.add(obj['image_id'])
    
    return list(analyzed - removed)[0:n_of_img]
    

# COSTANTI
ANNOTATIONS_PATH = './annotations/person_keypoints_val2017.json'
DS_PATH = './dataset/'
EFFICIENTPOSE_PATH = '../EfficientPose-master/'
EFFICIENTPOSE_MAIN = 'track.py'
MODELs = ['RT','I','II', 'III', 'IV']

# VARIABILI
FRAMEWORK = 'tflite' #, 'keras' #"torch"# "tf", #
MODEL = MODELs[0]
NUMBER_OF_IMAGES = 100

DOWNLOAD_IMAGES = False
SINGLE_PERSON = False

# True lavora su un solo modello (MODEL), False crea un plot che compara i diversi modelli
ANALYZE_ONE_MODEL = False
# variabile usata quando ANALYZE_ONE_MODEL=True. Questa variabile dice se devono essere creati 
# o meno i csv
CREATE_CSV_INFERENCE = False
# variabile usata quando ANALYZE_ONE_MODEL=False. Questa variabile dice quale metrica impiegare
# per la comparazione dei modelli
METRIC_NAME = 'pck'

# MAIN
if __name__ == "__main__":
    coco=COCO(ANNOTATIONS_PATH)
    person_cat_id = coco.getCatIds(catNms=['person'])[0]
    img_ids = coco.getImgIds(catIds=[person_cat_id])
    annotations = coco.loadAnns(ids=coco.getAnnIds(imgIds=img_ids))
    img_ids = search_person_image(annotations, NUMBER_OF_IMAGES, SINGLE_PERSON)

    # creo una mappa id-img e nome del file così da essere uniforme al codice già usato per MPII
    map_id_filename = {img['id'] : img['file_name'] for img in coco.loadImgs(img_ids)}
    # carico le annotazioni di interesse cioè quelle che rispettano i requisiti di search_person_image
    annotations = coco.loadAnns(ids=coco.getAnnIds(imgIds=img_ids))
    abs_ds_path = os.path.abspath(DS_PATH)

    if DOWNLOAD_IMAGES:
        coco.download(imgIds=img_ids, tarDir=abs_ds_path)
    
    # vengono creati i csv se necessario
    if ANALYZE_ONE_MODEL:
        if CREATE_CSV_INFERENCE:
            create_csv_model_inference(abs_ds_path, MODEL)
            move_csv_in_model_dir(abs_ds_path, MODEL)
    

        ground_truth = get_rows_from_annotations(annotations, map_id_filename)
        inference = get_inference(ground_truth.keys(), abs_ds_path, MODEL)    
        
        print("PCK:", utils.pck(ground_truth, inference))
        print("PCP:", utils.pcp(ground_truth, inference))
        print("PDJ:", utils.pdj(ground_truth, inference))
        print("AUC per PCK:", utils.auc(utils.pck, ground_truth, inference, _max=1, visualize=True, model_name=MODEL))
        print("AUC per PCP:", utils.auc(utils.pcp, ground_truth, inference, _max=1, visualize=True, model_name=MODEL))
        print("AUC per PDJ:", utils.auc(utils.pdj, ground_truth, inference, _max=1, visualize=True, model_name=MODEL))
    else:
        if CREATE_CSV_INFERENCE:
            for m in MODELs:
                create_csv_model_inference(abs_ds_path, m)
                move_csv_in_model_dir(abs_ds_path, m)

        compararison = compare_models(annotations, abs_ds_path, METRIC_NAME, map_id_filename)

        for m in MODELs:
            utils.auc_sup(compararison[m][0],compararison[m][1], m, METRIC_NAME)
            