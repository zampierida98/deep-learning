'''
Il programma prende i modelli messi a disposizione da EfficientPose-master per testare la bontà dei
modelli sul dataset MPII. L'insieme delle funzioni di supporto come la metrica impiegata viene scritto
in un file ausiliario così da poterla usare in un momento successivo
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
# dataset path
#DS_PATH = './MPII/images'
DS_PATH = './MPII/sub_dataset'
EFFICIENTPOSE_PATH = './EfficientPose-master/'
EFFICIENTPOSE_MAIN = 'track.py'
MODEL = ['RT', 'RT_Lite','I','I_Lite', 'II','II_Lite', 'III', 'IV'][7]
JOINT_ID = {0:"right_ankle", 1:"right_knee", 2: "right_hip", 3: "left_hip", 4: "left_knee", 5: "left_ankle", 6: "pelvis", 7:"thorax", 8:"upper_neck", 9:"head_top", 10:"right_wrist", 11:"right_elbow", 12:"right_shoulder", 13:"left_shoulder", 14:"left_elbow", 15: "left_wrist"}
# VARIABILI
annotations = None
create_csv_model_infer = False

# FUNZIONI
def get_rows_from_annotations(annotations, abs_ds_path):
    res = {}
    for name in os.listdir(abs_ds_path):
        if name.split(".")[1] == "csv":
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

def get_rows_from_csv(list_of_file, abs_ds_path):
    res = {}
    for file in list_of_file:
        with open(f'{abs_ds_path}\\{file.split(".")[0]}_coordinates.csv', 'r') as fin:
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

# MAIN
if __name__ == "__main__":
    with open('annotations.pickle', 'rb') as fin:
        annotations = pickle.load(fin)
    
    # path assoluto della dir DS_PATH
    abs_ds_path = os.path.abspath(DS_PATH)    

    # vengono creati i csv se necessario
    if create_csv_model_infer:
        list_of_file = []
        for file in os.listdir(abs_ds_path):
            if not file.endswith(".csv"):
                list_of_file.append(f'{abs_ds_path}\\{file}')
                
        os.chdir(EFFICIENTPOSE_PATH)

        for file in list_of_file:
            os.system(f'python {EFFICIENTPOSE_MAIN} --model={MODEL} --path="{file}" --store')
    
    # descrizione della struttura link: http://human-pose.mpi-inf.mpg.de/#download
    # Non per tutte le img abbiamo le predizioni.
    # TODO: è da investigare su questo fatto

    ground_truth = get_rows_from_annotations(annotations, abs_ds_path)
    # alcune img non hanno il campo annopoints per cui tali img non verranno recuperate da inference
    
    inference = get_rows_from_csv(ground_truth.keys(), abs_ds_path) #ground_truth.keys()
    inference = get_real_body_parts(inference, abs_ds_path)

    print("PCK:", utils.pckh(ground_truth, inference))
    print("PCP:", utils.pcp(ground_truth, inference))
    print("PDJ:", utils.pdj(ground_truth, inference))
    print("AUC per PCK:", utils.auc(utils.pckh, ground_truth, inference, _max=1, visualize=True))
    print("AUC per PCP:", utils.auc(utils.pcp, ground_truth, inference, _max=1, visualize=True))
    print("AUC per PDJ:", utils.auc(utils.pdj, ground_truth, inference, _max=1, visualize=True))
