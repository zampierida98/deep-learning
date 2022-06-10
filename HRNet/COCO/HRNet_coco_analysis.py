'''
Calcola le metriche a partire dai risultati del modello HRNet. utilizzando le consuete metriche presenti nel file
utils.py
'''
# IMPORT
import os
import numpy as np
import utils
from coco import COCO
import json

import warnings
warnings.simplefilter("ignore", FutureWarning)

# FUNZIONI

def map_points_into_standard_version(data):
    body_part_ids = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder",
                    "left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee",
                    "right_knee","left_ankle","right_ankle"]

    res = {}
    for obj in data:
        parts = {}
        # si lavora da 15 in quanto è corrispondente ai valori di left_shoulder
        for i in range(15,len(obj['keypoints']),3):
            parts[body_part_ids[i // 3]] = (round(obj['keypoints'][i]), round(obj['keypoints'][i+1]))        
                
        # bacino è collocato a metà tra le due anche
        if ((parts['left_hip'][0] == 0 and parts['left_hip'][1] == 0) or 
            (parts['right_hip'][0] == 0 and parts['right_hip'][1] == 0)):
            parts['pelvis'] = (0,0)
        else:
            parts['pelvis'] = ( round( (parts['left_hip'][0] + parts['right_hip'][0]) // 2),
                                round( (parts['left_hip'][1] + parts['right_hip'][1]) // 2)
                                )
        
        # il torace è un po' più in basso del valor medio tra i due 
        if ((parts['left_shoulder'][0] == 0 and parts['left_shoulder'][1] == 0) or 
            (parts['right_shoulder'][0] == 0 and parts['right_shoulder'][1] == 0)):
            parts['thorax'] = (0,0)
        else:
            parts['thorax'] = ( round( (parts['left_shoulder'][0] + parts['right_shoulder'][0]) // 2),
                                round( (parts['left_shoulder'][1] + parts['right_shoulder'][1]) // 2)
                                )
        
        # distanza sulla x tra i due occhi è circa uguale alla distanza centro_tra_i_due_occhi e la parte_superiore_testa
        if ((obj['keypoints'][3] == 0 and obj['keypoints'][4] == 0) or
            (obj['keypoints'][6] == 0 and obj['keypoints'][7] == 0) or
            (obj['keypoints'][0] == 0 and obj['keypoints'][1] == 1)):
            parts['head_top'] = (0,0)
        else:
            parts['head_top'] = (round( (obj['keypoints'][3] + obj['keypoints'][6]) // 2),
                                round( (obj['keypoints'][4] + obj['keypoints'][7]) // 2)
                                )
            distanza_occhi = abs(round(obj['keypoints'][6] - obj['keypoints'][3]))
        
            parts['head_top'] = (parts['head_top'][0] - distanza_occhi, parts['head_top'][1])        

        # upper_neck è collocato a metà tra naso e torace
        if ((parts['thorax'][0] == 0 and parts['thorax'][1] == 0) or 
            (obj['keypoints'][0] == 0 and obj['keypoints'][1] == 1)):
            parts['thorax'] = (0,0)
        else:
            parts['upper_neck'] = ( round( (obj['keypoints'][0] + parts['thorax'][0]) // 2),
                                round( (obj['keypoints'][1] + parts['thorax'][1]) // 2)
                                )
        
        res[obj['image_id'] ] = parts

    return res

def prettify(_dict, model_name, metric_name):
    print("#"*20, "MODEL " + model_name.upper(), metric_name.upper(), "#"*20)
    for k in _dict:
        print("\t", k, "\t\t", round(_dict[k]*100, 1), "%")
    print("#"*54)

# COSTANTI

# VARIABILI
HRNET_RESULT_PATH = './results/keypoints_val2017_results_0.json'
ANNOTATIONS_PATH = '../../COCO/annotations/person_keypoints_val2017.json'
TAU_FOR_PRETTIFY = 0.5
MODEL='HRNET_W32_256x256'

# MAIN
if __name__ == "__main__":
    # carico i risultati di HRNet da json
    with open(HRNET_RESULT_PATH, 'r') as fin:
        hrnet_data = json.load(fin)
    
    # recuperiamo i seguenti gli img_id dal file prodotto da HRNet
    img_ids = []
    for obj in hrnet_data:
        img_ids.append(obj['image_id'])

    coco=COCO(ANNOTATIONS_PATH)
    # carico le annotazioni di interesse cioè quelle che rispettano i requisiti di search_person_image
    annotations = coco.loadAnns(ids=coco.getAnnIds(imgIds=img_ids))

    ground_truth = map_points_into_standard_version(annotations)
    inference = map_points_into_standard_version(hrnet_data)
    
    # [0] per il recupero del solo valore della metrica
    print("PCK:", utils.pck(ground_truth, inference)[0])
    prettify(utils.pck(ground_truth, inference, TAU_FOR_PRETTIFY)[1], MODEL, "pck")
    print("PCP:", utils.pcp(ground_truth, inference)[0])
    prettify(utils.pcp(ground_truth, inference, TAU_FOR_PRETTIFY)[1], MODEL, "pcp")
    print("PDJ:", utils.pdj(ground_truth, inference)[0])
    prettify(utils.pdj(ground_truth, inference, TAU_FOR_PRETTIFY)[1], MODEL, "pdj")

    print("AUC per PCK:", utils.auc(utils.pck, ground_truth, inference, _max=1, visualize=True, model_name=MODEL))
    print("AUC per PCP:", utils.auc(utils.pcp, ground_truth, inference, _max=1, visualize=True, model_name=MODEL))
    print("AUC per PDJ:", utils.auc(utils.pdj, ground_truth, inference, _max=1, visualize=True, model_name=MODEL))