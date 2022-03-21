# -*- coding: utf-8 -*-
'''
Il programma prende i modelli messi a disposizione da EfficientPose per testarne la bontà sul dataset MPII.
L'insieme delle funzioni di supporto (ad es. metriche impiegate) viene scritto in un file ausiliario così da poterle usare in un momento successivo
'''

# IMPORT
import os
import utils
import warnings
from itertools import groupby
from PIL import Image
warnings.simplefilter("ignore", FutureWarning)

# COSTANTI
DS_PATH = './LSP/results'
MODELS = ['RT','I','II','III','IV']
JOINT_ID = {0:"right_ankle", 1:"right_knee", 2:"right_hip", 3:"left_hip",
            4:"left_knee", 5:"left_ankle", 6:"pelvis", 7:"thorax",
            8:"upper_neck", 9:"head_top", 10:"right_wrist", 11:"right_elbow", 
            12:"right_shoulder", 13:"left_shoulder", 14:"left_elbow", 15: "left_wrist"}
JOINT_ID_LSP = {0:"right_ankle", 1:"right_knee", 2:"right_hip", 3:"left_hip",
                4:"left_knee", 5:"left_ankle", 6:"right_wrist", 7:"right_elbow",
                8:"right_shoulder", 9:"left_shoulder", 10:"left_elbow",
                11:"left_wrist", 12:"neck", 13:"head_top"}
FRAMEWORK = 'pytorch'
MODEL = MODELS[0]

# VARIABILI
# True lavora su un solo modello (MODEL), False crea un plot che compara i diversi modelli
one_or_more = True
# variabile che dice quale metrica impiegare per la comparazione dei modelli
metric_name = 'pckh'

# FUNZIONI
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
        
        # TODO: upper-neck!=thorax E QUINDI neck=punto medio tra upper-neck e thorax
        inference[img]['neck'] = (round((inference[img][JOINT_ID[7]][0]+inference[img][JOINT_ID[8]][0]) / 2),
                                  round((inference[img][JOINT_ID[7]][1]+inference[img][JOINT_ID[8]][1]) / 2))
        # TODO: pelvis = punto medio tra le anche? 1/9 E 0/9 CON RT
        
    return inference

def get_inference(list_of_img, abs_ds_path, model):
    # alcune img non hanno il campo annopoints per cui tali img non verranno recuperate da inference
    inference = get_rows_from_csv(list_of_img, abs_ds_path, model)
    return get_real_body_parts(inference, abs_ds_path)

# LSP - http://sam.johnson.io/research/lsp.html
def load_LSP():
    mat = utils.load_mat('./LSP/joints.mat')
    joints = mat['joints']  # 3x14x2000
    
    annotations = {}
    for img in range(joints.shape[2]):  # 2000 immagini
        ann = {}
        for kp in range(joints.shape[1]):  # 14 keypoint
            #Right ankle,Right knee,Right hip,Left hip,Left knee,Left ankle,Right wrist,
            #Right elbow,Right shoulder,Left shoulder,Left elbow,Left wrist,Neck,Head top
            ann[JOINT_ID_LSP[kp]] = (round(joints[0][kp][img]), round(joints[1][kp][img]))
        # TODO: pelvis = punto medio tra le anche? 1/9 E 0/9 CON RT
        ann["pelvis"] = (round((ann[JOINT_ID_LSP[2]][0]+ann[JOINT_ID_LSP[3]][0]) / 2),
                         round((ann[JOINT_ID_LSP[2]][1]+ann[JOINT_ID_LSP[3]][1]) / 2))
        
        annotations[f'im{(img+1):04}.jpg'] = ann
    
    return annotations

# METRICHE
def distance(x1, y1, x2, y2):
    return ((x2-x1)**2 + (y2-y1)**2)**(1/2)

def get_segments():
    return [('head_top', 'neck'), ('neck', 'right_shoulder'), ('neck', 'left_shoulder'), 
            ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
            ('neck', 'pelvis'), ('pelvis', 'right_hip'), ('pelvis', 'left_hip'),
            ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
            ('left_hip', 'left_knee'), ('left_knee', 'left_ankle')]

def pcp(ground_truth, inference, tau=0.5):
    segments = get_segments()
    res = {'total': 0}
    for bp1, bp2 in segments:
        counter = 0  # ci sono immagini in cui lo scheletro potrebbe essere parzialmente osservabile
        correct_pred = 0
        
        for img in ground_truth:
            try:
                bp1x1, bp1y1 = inference[img][bp1]
                bp2x1, bp2y1 = inference[img][bp2]
                bp1x2, bp1y2 = ground_truth[img][bp1]
                bp2x2, bp2y2 = ground_truth[img][bp2]
                if (distance(bp1x1, bp1y1, bp1x2, bp1y2) <= tau * distance(bp1x2, bp1y2, bp2x2, bp2y2) and 
                    distance(bp2x1, bp2y1, bp2x2, bp2y2) <= tau * distance(bp1x2, bp1y2, bp2x2, bp2y2)):

                    correct_pred += 1
                counter += 1
            except:
                pass
        
        # se il segmento non esiste non lo conto
        try:
            res[(bp1, bp2)] = correct_pred / counter
            res['total'] += correct_pred / counter
        except:
            pass

    res['total'] /= (len(res.keys())-1)
    return res

def pdj(ground_truth, inference, tau=0.5):
    segments = get_segments()
    res = {'total': 0}
    for bp1, bp2 in segments:
        counter = 0  # ci sono immagini in cui lo scheletro potrebbe essere parzialmente osservabile
        correct_pred = 0
        
        for img in ground_truth:
            try:
                torso_diag = max(distance(ground_truth[img]['right_shoulder'][0], ground_truth[img]['right_shoulder'][1], 
                                          ground_truth[img]['left_hip'][0], ground_truth[img]['left_hip'][1]), 
                                distance(ground_truth[img]['left_shoulder'][0], ground_truth[img]['left_shoulder'][1], 
                                          ground_truth[img]['right_hip'][0], ground_truth[img]['right_hip'][1]))

                bp1x1, bp1y1 = inference[img][bp1]
                bp2x1, bp2y1 = inference[img][bp2]
                bp1x2, bp1y2 = ground_truth[img][bp1]
                bp2x2, bp2y2 = ground_truth[img][bp2]
                if (distance(bp1x1, bp1y1, bp1x2, bp1y2) <= tau * torso_diag and 
                    distance(bp2x1, bp2y1, bp2x2, bp2y2) <= tau * torso_diag):

                    correct_pred += 1
                counter += 1
            except:
                pass
        
        # se il segmento non esiste non lo conto
        try:
            res[(bp1, bp2)] = correct_pred / counter
            res['total'] += correct_pred / counter
        except:
            pass
    
    res['total'] /= (len(res.keys())-1)
    return res

def pck(ground_truth, inference, tau=0.5):
    res = {'total': 0}
    for k, p in JOINT_ID_LSP.items():
        counter = 0  # ci sono immagini in cui lo scheletro potrebbe essere parzialmente osservabile
        correct_pred = 0
        
        for img in ground_truth:
            try:
                torso_diag = max(distance(ground_truth[img]['right_shoulder'][0], ground_truth[img]['right_shoulder'][1], 
                                          ground_truth[img]['left_hip'][0], ground_truth[img]['left_hip'][1]), 
                                distance(ground_truth[img]['left_shoulder'][0], ground_truth[img]['left_shoulder'][1], 
                                          ground_truth[img]['right_hip'][0], ground_truth[img]['right_hip'][1]))

                x1, y1 = inference[img][p]
                x2, y2 = ground_truth[img][p]
                if distance(x1, y1, x2, y2) <= tau * torso_diag:
                    correct_pred += 1
                counter += 1
            except:
                pass
            
        # se il keypoint non esiste non lo conto
        try:
            res[p] = correct_pred / counter
            res['total'] += correct_pred / counter
        except:
            pass
        
    res['total'] /= (len(res.keys())-1)
    return res

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
        inference = get_inference(images, abs_ds_path, MODEL)
        
        # PCP totale e per Torso?,Upper Leg,Lower Leg,Upper Arm,Forearm,Head
        print("PCP:", pcp(ground_truth, inference))
        print()
        
        print("PDJ:", pdj(ground_truth, inference))
        print()
        
        # PCK con d=torso (PER PELVIS?)
        print("PCK:", pck(ground_truth, inference))
        print()

    else:
        pass

        #compararison = compare_models(annotations, abs_ds_path, metric_name)
        # metric = {'pckh':utils.pckh, 'pcp':utils.pcp, 'pdj':utils.pdj}
        # _min,_max,step = 0, 1, 0.01
        # comparison = {}
        # for m in MODELS:
        #     print("Lavoro su EfficientPose", m, "...")

        #     inference = get_inference(images, abs_ds_path, m)
        #     X = [i for i in np.arange(_min, _max, step)]
        #     Y = [metric[metric_name](ground_truth, inference, x) for x in X]
        #     comparison[m] = (X,Y)

        # utils.plot(comparison, metric_name)

        # for m in MODELS:
        #     utils.auc_2(comparison[m][0],comparison[m][1], m, metric_name)


if __name__ == "__main__":
    #main_MPII()
    main_LSP()
