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
MODELs = ['RT', 'RT_Lite','I','I_Lite', 'II','II_Lite', 'III', 'IV']
JOINT_ID = {0:"right_ankle", 1:"right_knee", 2: "right_hip", 3: "left_hip", 4: "left_knee", 5: "left_ankle", 
            6: "pelvis", 7:"thorax", 8:"upper_neck", 9:"head_top", 10:"right_wrist", 11:"right_elbow", 
            12:"right_shoulder", 13:"left_shoulder", 14:"left_elbow", 15: "left_wrist"}
# VARIABILI
MODEL = MODELs[7]
# True lavora su un solo modello (MODEL), False crea un plot che compara i diversi modelli
one_or_more = True
# variabile usata quando one_or_more=True. Questa variabile dice se devono essere creati 
# o meno i csv
create_csv_model_infer = False
# variabile usata quando one_or_more=False. Questa variabile dice quale metrica impiegare
# per la comparazione dei modelli
metric_name = 'pckh'

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
        os.system(f'python {EFFICIENTPOSE_MAIN} --model={model} --path="{file}" --store')

def get_inference(list_of_img, abs_ds_path):
    # alcune img non hanno il campo annopoints per cui tali img non verranno recuperate da inference
    inference = get_rows_from_csv(list_of_img, abs_ds_path)
    return get_real_body_parts(inference, abs_ds_path)

def compare_models(annotations, abs_ds_path, metric_name, for_auc=False):
    metric = {'pck': utils.pckh, 'pcp':utils.pcp, 'pdj':utils.pdj}
    _min,_max,step = 0, 1, 0.01
    res = {}
    ground_truth = get_rows_from_annotations(annotations, abs_ds_path)
    for m in MODELs:
        print("Lavoro su EfficientPose", m, "...\n")

        create_csv_model_inference(abs_ds_path, m)
        inference = get_inference(ground_truth.keys(), abs_ds_path)
        X = [i for i in np.arange(_min, _max, step)]
        if for_auc:
            Y = [utils.auc(metric[metric_name], ground_truth, inference, _max=x) for x in X]
        else:
            Y = [metric[metric_name](ground_truth, inference, x) for x in X]
        res[m] = (X,Y)
        
        print()
    utils.plot(res, (metric_name if not for_auc else "auc per"+metric_name))

# MAIN
if __name__ == "__main__":
    with open('annotations.pickle', 'rb') as fin:
        annotations = pickle.load(fin)
        
    # path assoluto della dir DS_PATH
    abs_ds_path = os.path.abspath(DS_PATH)    

    # vengono creati i csv se necessario
    if one_or_more:
        if create_csv_model_infer:
            create_csv_model_inference(abs_ds_path, MODEL)
    
        # descrizione della struttura link: http://human-pose.mpi-inf.mpg.de/#downloa

        ground_truth = get_rows_from_annotations(annotations, abs_ds_path)
        inference = get_inference(annotations, abs_ds_path)    

        print("PCK:", utils.pckh(ground_truth, inference))
        print("PCP:", utils.pcp(ground_truth, inference))
        print("PDJ:", utils.pdj(ground_truth, inference))
        print("AUC per PCK:", utils.auc(utils.pckh, ground_truth, inference, _max=1, visualize=True, model_name=MODEL))
        print("AUC per PCP:", utils.auc(utils.pcp, ground_truth, inference, _max=1, visualize=True, model_name=MODEL))
        print("AUC per PDJ:", utils.auc(utils.pdj, ground_truth, inference, _max=1, visualize=True, model_name=MODEL))
    else:
        compare_models(annotations, abs_ds_path, metric_name, for_auc=False)
