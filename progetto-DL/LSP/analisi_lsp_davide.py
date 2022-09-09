# -*- coding: utf-8 -*-
# IMPORT
import os
import utils
import warnings
from itertools import groupby
from PIL import Image
warnings.simplefilter("ignore", FutureWarning)

# COSTANTI
CSV_PATH = './results'
MODELS = ['RT','I','II','III','IV']
FRAMEWORK = 'pytorch'
MODEL = MODELS[0]
JOINT_ID = {0:"right_ankle", 1:"right_knee", 2:"right_hip", 3:"left_hip",
            4:"left_knee", 5:"left_ankle", 6:"pelvis", 7:"thorax",
            8:"upper_neck", 9:"head_top", 10:"right_wrist", 11:"right_elbow", 
            12:"right_shoulder", 13:"left_shoulder", 14:"left_elbow", 15: "left_wrist"}
JOINT_ID_LSP = {0:"right_ankle", 1:"right_knee", 2:"right_hip", 3:"left_hip",
                4:"left_knee", 5:"left_ankle", 6:"right_wrist", 7:"right_elbow",
                8:"right_shoulder", 9:"left_shoulder", 10:"left_elbow",
                11:"left_wrist", 12:"neck", 13:"head_top"}

# VARIABILI
# True lavora su un solo modello (MODEL), False crea un plot che compara i diversi modelli
ANALYZE_ONE_MODEL = False

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
        
        # upper-neck!=thorax E QUINDI neck=punto medio tra upper-neck e thorax
        inference[img]['neck'] = (round((inference[img][JOINT_ID[7]][0]+inference[img][JOINT_ID[8]][0]) / 2),
                                  round((inference[img][JOINT_ID[7]][1]+inference[img][JOINT_ID[8]][1]) / 2))
        # TODO: pelvis = punto medio tra le anche? 1/9 E 0/9 CON RT
        
    return inference

def get_inference(list_of_img, abs_ds_path, model):
    inference = get_rows_from_csv(list_of_img, abs_ds_path, model)
    return get_real_body_parts(inference, abs_ds_path)

# LSP - http://sam.johnson.io/research/lsp.html
def load_LSP():
    mat = utils.load_mat('./joints.mat')
    joints = mat['joints']  # 3x14x2000
    
    annotations = {}
    for img in range(joints.shape[2]):  # 2000 immagini
        ann = {}
        for kp in range(joints.shape[1]):  # 14 keypoint
            #Right ankle,Right knee,Right hip,Left hip,Left knee,Left ankle,Right wrist,
            #Right elbow,Right shoulder,Left shoulder,Left elbow,Left wrist,Neck,Head top
            ann[JOINT_ID_LSP[kp]] = (round(joints[0][kp][img]), round(joints[1][kp][img]))
        
        # pelvis=punto medio tra le anche
        ann["pelvis"] = (round((ann[JOINT_ID_LSP[2]][0]+ann[JOINT_ID_LSP[3]][0]) / 2),
                         round((ann[JOINT_ID_LSP[2]][1]+ann[JOINT_ID_LSP[3]][1]) / 2))
        
        annotations[f'im{(img+1):04}.jpg'] = ann
    
    return annotations

def main_LSP():
    # path assoluto della cartella CSV_PATH
    abs_ds_path = os.path.abspath(CSV_PATH)
    
    # lista con i nomi delle immagini per get_inference
    images = [ f for f in os.listdir(abs_ds_path) if os.path.isfile(os.path.join(abs_ds_path,f)) ]

    # ground truth
    annotations = load_LSP()
    ground_truth = {}
    for name in images:
        ground_truth[name] = annotations[name]

    if ANALYZE_ONE_MODEL:
        inference = get_inference(images, abs_ds_path, MODEL)
        
        # PCP totale e per Torso?,Upper Leg,Lower Leg,Upper Arm,Forearm,Head
        print("PCP:", utils.pcp(ground_truth, inference))
        pcp_values = utils.auc(utils.pcp, ground_truth, inference)
        print("AUC per PCP: ", pcp_values[2])
        print()
        
        print("PDJ:", utils.pdj(ground_truth, inference))
        pdj_values = utils.auc(utils.pdj, ground_truth, inference)
        print("AUC per PDJ: ", pdj_values[2])
        print()
        
        # PCK con d=torso (PER PELVIS?)
        print("PCK:", utils.pck(ground_truth, inference))
        pck_values = utils.auc(utils.pck, ground_truth, inference)
        print("AUC per PCK: ", pck_values[2])
        print()
        
        # grafici comparativi
        utils.plot(utils.pcp, {MODEL: pcp_values})
        utils.plot(utils.pdj, {MODEL: pdj_values})
        utils.plot(utils.pck, {MODEL: pck_values})

    else:
        pcps = {}
        pdjs = {}
        pcks = {}
        for m in MODELS:
            inference = get_inference(images, abs_ds_path, m)
            
            pcp_values = utils.auc(utils.pcp, ground_truth, inference)
            pcps[m] = pcp_values
            
            pdj_values = utils.auc(utils.pdj, ground_truth, inference)
            pdjs[m] = pdj_values
            
            pck_values = utils.auc(utils.pck, ground_truth, inference)
            pcks[m] = pck_values
            
            print("Modello "+m)
            print("AUC per PCP: ", pcp_values[2])
            print("AUC per PDJ: ", pdj_values[2])
            print("AUC per PCK: ", pck_values[2])
            print()

        # grafici comparativi
        utils.plot(utils.pcp, pcps)
        utils.plot(utils.pdj, pdjs)
        utils.plot(utils.pck, pcks)


if __name__ == "__main__":
    main_LSP()
