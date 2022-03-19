# -*- coding: utf-8 -*-
'''
Il programma prende i modelli messi a disposizione da EfficientPose per testarne la bontà sul dataset MPII.
L'insieme delle funzioni di supporto (ad es. metriche impiegate) viene scritto in un file ausiliario così da poterle usare in un momento successivo
'''

# IMPORT
import pickle
import os
import utils
import numpy as np
import warnings
from itertools import groupby
from PIL import Image
warnings.simplefilter("ignore", FutureWarning)

# COSTANTI
DS_PATH = './LSP/prova'
EFFICIENTPOSE_PATH = './EfficientPose-master/'
EFFICIENTPOSE_MAIN = 'track.py'
MODELS = ['RT','I','II','III','IV']
JOINT_ID = {0:"right_ankle", 1:"right_knee", 2: "right_hip", 3: "left_hip", 4: "left_knee", 5: "left_ankle", 
            6: "pelvis", 7:"thorax", 8:"upper_neck", 9:"head_top", 10:"right_wrist", 11:"right_elbow", 
            12:"right_shoulder", 13:"left_shoulder", 14:"left_elbow", 15: "left_wrist"}
FRAMEWORK = 'pytorch'
MODEL = MODELS[0]

# VARIABILI
# True lavora su un solo modello (MODEL), False crea un plot che compara i diversi modelli
one_or_more = True
# variabile che dice se devono essere creati o meno i csv
create_csv_model_infer = False
# variabile che dice quale metrica impiegare per la comparazione dei modelli
metric_name = 'pckh'

# FUNZIONI
def get_rows_from_annotations(annotations, abs_ds_path):
    res = {}
    for name in os.listdir(abs_ds_path):
        if name.endswith(".csv"):
            continue
        
        try:
            row = annotations[name]
            
            parts = {}
            parts['head_box'] = (row['annorect']['x1'],
                                 row['annorect']['y1'],
                                 row['annorect']['x2'],
                                 row['annorect']['y2'])
            for d in row['annorect']['annopoints']['point']: # 16 parti del corpo
                parts[JOINT_ID[d['id']]] = (round(d['x']), round(d['y']))
            res[name] = parts
        except:
            pass

    return res

def get_rows_from_csv(list_of_file, abs_ds_path, model):
    res = {}
    for file in list_of_file:
        with open(f'{abs_ds_path}/{model}/{file.split(".")[0]}_coordinates.csv', 'r') as fin:
            header = fin.readline().strip()
            header = header.replace("_x", ":x")
            header = header.replace("_y", ":y")

            if FRAMEWORK != 'pytorch':
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
        image = Image.open(f'{abs_ds_path}/{img}')
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
            list_of_file.append(f'{abs_ds_path}/{file}')
    
    # se il processo non si trova già in EFFICIENTPOSE_PATH viene fatto il cambio di directory
    try:
        os.chdir(EFFICIENTPOSE_PATH)
    except:
        pass

    # per ogni file viene creato il csv
    for file in list_of_file:
        os.system(f'python3.7 {EFFICIENTPOSE_MAIN} --model={model} --path="{file}" --store --framework={FRAMEWORK}')

def get_inference(list_of_img, abs_ds_path, model):
    # alcune img non hanno il campo annopoints per cui tali img non verranno recuperate da inference
    inference = get_rows_from_csv(list_of_img, abs_ds_path, model)
    return get_real_body_parts(inference, abs_ds_path)

def compare_models(annotations, abs_ds_path, metric_name):
    metric = {'pckh': utils.pckh, 'pcp':utils.pcp, 'pdj':utils.pdj}
    _min,_max,step = 0, 1, 0.01
    res = {}
    ground_truth = get_rows_from_annotations(annotations, abs_ds_path)
    for m in MODELS:
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
        os.mkdir(f'{abs_ds_path}/{model}')
    except:
        pass
    
    for file in os.listdir(abs_ds_path):
        if file.endswith(".csv"):
            os.replace(f'{abs_ds_path}/{file}', f'{abs_ds_path}/{model}/{file}')

# MPII - http://human-pose.mpi-inf.mpg.de/
def main_MPII():
    with open('annotations.pickle', 'rb') as fin:
        annotations = pickle.load(fin)
        
    # path assoluto della dir DS_PATH
    abs_ds_path = os.path.abspath(DS_PATH)

    # vengono creati i csv se necessario
    if one_or_more:
        if create_csv_model_infer:
            create_csv_model_inference(abs_ds_path, MODEL)
            move_csv_in_model_dir(abs_ds_path, MODEL)

        ground_truth = get_rows_from_annotations(annotations, abs_ds_path)
        inference = get_inference(ground_truth.keys(), abs_ds_path, MODEL)    

        print("PCK:", utils.pckh(ground_truth, inference))
        print("PCP:", utils.pcp(ground_truth, inference))
        print("PDJ:", utils.pdj(ground_truth, inference))
        print("AUC per PCK:", utils.auc(utils.pckh, ground_truth, inference, _max=1, visualize=True, model_name=MODEL))
        print("AUC per PCP:", utils.auc(utils.pcp, ground_truth, inference, _max=1, visualize=True, model_name=MODEL))
        print("AUC per PDJ:", utils.auc(utils.pdj, ground_truth, inference, _max=1, visualize=True, model_name=MODEL))
    else:
        if create_csv_model_infer:
            for m in MODELS:
                create_csv_model_inference(abs_ds_path, m)
                move_csv_in_model_dir(abs_ds_path, m)

        compararison = compare_models(annotations, abs_ds_path, metric_name)

        for m in MODELS:
            utils.auc_2(compararison[m][0],compararison[m][1], m, metric_name)

# LSP - http://sam.johnson.io/research/lsp.html
def load_LSP():
    mat = utils.load_mat('./LSP/joints.mat')
    joints = mat['joints']  # 3x14x2000
    JOINT_ID_LSP = {0:"right_ankle", 1:"right_knee", 2:"right_hip", 3:"left_hip",
                    4:"left_knee", 5:"left_ankle", 6:"right_wrist", 7:"right_elbow",
                    8:"right_shoulder", 9:"left_shoulder", 10:"left_elbow",
                    11:"left_wrist", 12:"upper_neck", 13:"head_top"}
    
    annotations = {}
    for img in range(joints.shape[2]):  # 2000 immagini
        ann = {}
        for kp in range(joints.shape[1]):  # 14 keypoint
            #Right ankle,Right knee,Right hip,Left hip,Left knee,Left ankle,Right wrist,
            #Right elbow,Right shoulder,Left shoulder,Left elbow,Left wrist,Neck,Head top
            ann[JOINT_ID_LSP[kp]] = (round(joints[0][kp][img]), round(joints[1][kp][img]))
        # TODO: punto medio tra le anche=pelvis?
        annotations[f'im{(img+1):04}.jpg'] = ann
    
    return annotations

def main_LSP():
    # path assoluto della dir DS_PATH
    abs_ds_path = os.path.abspath(DS_PATH)
    
    # lista con i nomi delle immagini per get_inference
    images = [ f for f in os.listdir(abs_ds_path) if os.path.isfile(os.path.join(abs_ds_path,f)) ]

    # ground truth
    annotations = load_LSP()
    ground_truth = {}
    for name in images:
        ground_truth[name] = annotations[name]
    
    # vengono creati i csv se necessario
    if one_or_more:
        if create_csv_model_infer:
            create_csv_model_inference(abs_ds_path, MODEL)
            move_csv_in_model_dir(abs_ds_path, MODEL)

        inference = get_inference(images, abs_ds_path, MODEL)
        
        print("PCP Total:", utils.pcp(ground_truth, inference))
        print("AUC per PCP Total:", utils.auc(utils.pcp, ground_truth, inference, _max=1, visualize=True, model_name=MODEL))
        # TODO: PCP per Torso,Upper Leg,Lower Leg,Upper Arm,Forearm,Head
    else:
        if create_csv_model_infer:
            for m in MODELS:
                create_csv_model_inference(abs_ds_path, m)
                move_csv_in_model_dir(abs_ds_path, m)

        #compararison = compare_models(annotations, abs_ds_path, metric_name)
        metric = {'pckh':utils.pckh, 'pcp':utils.pcp, 'pdj':utils.pdj}
        _min,_max,step = 0, 1, 0.01
        comparison = {}
        for m in MODELS:
            print("Lavoro su EfficientPose", m, "...")

            inference = get_inference(images, abs_ds_path, m)
            X = [i for i in np.arange(_min, _max, step)]
            Y = [metric[metric_name](ground_truth, inference, x) for x in X]
            comparison[m] = (X,Y)

        utils.plot(comparison, metric_name)

        for m in MODELS:
            utils.auc_2(comparison[m][0],comparison[m][1], m, metric_name)


if __name__ == "__main__":
    #main_MPII()
    main_LSP()
