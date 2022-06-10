# -*- coding: utf-8 -*-
# IMPORT
import pickle
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

# VARIABILI
# True lavora su un solo modello (MODEL), False crea un plot che compara i diversi modelli
ANALYZE_ONE_MODEL = False


# FUNZIONI PER EFFICIENT POSE
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

def get_inference(list_of_img, abs_ds_path, model):
    inference = get_rows_from_csv(list_of_img, abs_ds_path, model)
    return get_real_body_parts(inference, abs_ds_path)

# FUNZIONI PER HRNET
def load_HRNet_preds():
    mat = utils.load_mat('./pred.mat')
    joints = mat['preds']  # 2958x16x2

    f = open("img_names.log", "r")
    lines = f.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].strip()

    predictions = {}
    for img in range(joints.shape[0]):  # 2958 immagini
        p = {}
        for kp in range(joints.shape[1]):  # 16 keypoint
            p[JOINT_ID[kp]] = (round(joints[img][kp][0]), round(joints[img][kp][1]))

        predictions[lines[img].split('images/')[1]] = p
        # TODO: alcune immagini sono ripetute (tengo la prima o l'ultima?)

    return predictions

# FUNZIONI PER MPII
def get_rows_from_annotations(annotations, images):
    res = {}
    for name in images:
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


# MAIN
def main():
    # path assoluto della cartella CSV_PATH
    abs_ds_path = os.path.abspath(CSV_PATH)

    # predizioni di HRNet
    predictions = load_HRNet_preds()

    # lista con i nomi delle immagini per get_inference
    images = [ f for f in os.listdir(abs_ds_path) if os.path.isfile(os.path.join(abs_ds_path,f)) ]
    images = set(images) & set(predictions.keys())

    # sottoinsieme delle predizioni
    sub_pred = {}
    for name in images:
        sub_pred[name] = predictions[name]
        
    # annotazioni di MPII
    with open('../../annotations.pickle', 'rb') as fin:
        annotations = pickle.load(fin)

    ground_truth = get_rows_from_annotations(annotations, images)

    return 0

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
    main()
