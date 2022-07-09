# -*- coding: utf-8 -*-
'''
TEST QUANTITATIVO:  lo scopo è quello di prendere un dataset conosciuto e calcolarne le metriche sapendo
                    che prima verrà applicato il detector yolov5

INSERIRE IL SEGUENTE CODICE NELLA ROOT DI: https://github.com/ultralytics/yolov5
INSERIRE LE IMMAGINI NELLA DIRECTORY: DS_PATH
'''

# IMPORT
from cgitb import reset
from PIL import Image
import torch
#import torchvision
import pickle
import numpy as np
from itertools import groupby
import cv2
import os
import warnings
import utils_detector

warnings.simplefilter("ignore", FutureWarning)

# FUNZIONI
def get_rows_from_annotations(annotations, abs_ds_path):
    '''
    @annotations: dizionario python (trasformato da .mat a diz di strutture dati python) 
    @abs_ds_path: stringa del path assoluto del dataset di immagini
    
    Da annotations estrae le coordinate dei keypoint dell'immagine insieme al head-box (se presente)

    @return: dizionario del tipo {immagine: {parte_corpo: (x_pc, y_pc), ...} }
    '''
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
    '''
    @list_of_file: lista di stringhe contenenti i nomi dei file da analizzare
    @abs_ds_path: stringa del path assoluto delle immagini del dataset. In questo caso sarà la posizione
                  dei csv delle immagini cropped

    @model: stringa che indica il modello di EP

    Dai file csv generati dal modello di EP vengono estratte le coordinate dei keypoint dell'immagine

    @return: dizionario del tipo {immagine: {parte_corpo: (x_pc, y_pc), ...} }
    '''
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
    '''
    @inference: dizionario contenente l'inferenza fatta da EP sulle immagini del dataset. 
                E' del tipo {immagine: {parte_corpo: (x_pc, y_pc), ...} } con x_pc e y_pc nell'intervallo [0,1]

    @abs_ds_path: stringa del path assoluto delle immagini del dataset.

    @return:    stessa inferenza con i punti che adesso identificano uno specifico pixel. 
                x e y sono interi e non piu' nell'intervallo [0,1]
    '''
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
    '''
    @abs_ds_path: stringa del path assoluto delle immagini del dataset
    @model: stringa che indica il modello di EP

    Per ogni immagine del dataset viene generato un csv che è la stima della posa della persona dentro l'immagine

    @return: None
    '''
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
    '''
    @list_of_img: lista di stringhe contenenti i nomi dei file da analizzare
    @abs_ds_path: stringa del path assoluto delle immagini del dataset.
    @model: stringa che indica il modello di EP

    Genera l'oggetto 'inference' contenente le parti del corpo per ogni persona

    @return: dizionario del tipo {immagine: {parte_corpo: (x_pc, y_pc), ...} }
    '''
    # alcune img non hanno il campo annopoints per cui tali img non verranno recuperate da inference
    inference = get_rows_from_csv(list_of_img, abs_ds_path, model)
    return get_real_body_parts(inference, abs_ds_path)

def compare_models(annotations, abs_ds_path, metric_name, tau):
    metric = {'pckh': utils_detector.pckh, 'pcp':utils_detector.pcp, 'pdj':utils_detector.pdj}
    _min,_max,step = 0, 1, 0.01
    res = {}

    ground_truth = get_rows_from_annotations(annotations, abs_ds_path)
    # dato che il ground truth è riferito all'immagine completa e non al crop allora sistemo i punti
    ground_truth = normalize_ground_truth(ground_truth, top_left_coords)

    for m in MODELS:
        print("Lavoro su EfficientPose", m, "...")

        inference = get_inference(ground_truth.keys(), abs_ds_path, m)
        X = [i for i in np.arange(_min, _max, step)]
        Y = [metric[metric_name](ground_truth, inference, x) for x in X]
        # recupero solo il valore della metrica
        Y = [r1 for (r1,_) in Y]
        res[m] = (X,Y)
        # prettify per visualizzare dei risultati
        #print(metric_name, tau, ":", round(metric[metric_name](ground_truth, inference, tau)[0]*100, 2))
        prettify(metric[metric_name](ground_truth, inference, tau)[1], m, metric_name)

    utils_detector.plot(res, metric_name)
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
            
def prettify(_dict, model_name, metric_name):
    print("#"*20, "MODEL " + model_name.upper(), metric_name.upper(), "#"*20)
    for k in _dict:
        print("\t", k, "\t\t", round(_dict[k]*100, 1), "%")
    print("#"*54)

##################################################################################################################
##################################################################################################################
##################################################################################################################
def video_detection(video, model):
    res = {}
    #for ind, frame in enumerate(video):
        #res[ind] = model(frame['data']).print()
        
    vidcap = cv2.VideoCapture(video)
    success, image = vidcap.read()
    count = 0
    while success:
        res[count] = model(image)
        res[count].print()
        
        success, image = vidcap.read()
        count += 1
    
    return res

def detect():
    # Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom
    
    detections = {}
    video_detections = {}
    
    abs_path_dataset = os.path.abspath(DS_PATH)
    # Images
    for img in os.listdir(abs_path_dataset):

        # se img è una cartella allora chiaramente non la analizziamo
        if not os.path.isfile(os.path.join(abs_path_dataset, img)):
            continue

        if not (VIDEO_EXTENTION in img): # se sono immagini allora
            # Inference
            results = model(os.path.join(abs_path_dataset, img))
            
            # Results
            results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
            
            detections[img] = results
        else:
            # @Problema: Not compiled with video_reader support, to enable video_reader support, please install ffmpeg (version 4.2 is currently supported) and build torchvision from source. 
            #video_detections[img] = video_detection(torchvision.io.VideoReader(os.path.join(abs_path_dataset, img)), model)
            video_detections[img] = video_detection(os.path.join(abs_path_dataset, img), model)

    return detections, video_detections

def extract_people_in_images(detections):
    '''
    Prende come input le detection fatte da 'detect()'. Poi per ogni immagine determina le bounding box
    di tutte le persone nell'immagine. Applica infine crop che ritaglia dall'immagine la persona con il
    grado di confidenza più alto.

    '''
    # dizionario della forma - (img: crop_persona_con_piu_alta_confidenza)
    people = {}

    # https://docs.ultralytics.com/tutorials/pytorch-hub/#detailed-example
    for img in detections.keys():
        # predictions (pandas)
        bbox = detections[img].pandas().xyxy[0]  # xyxy=diagonale
        
        # vengono considerate solo le persone all'interno dell'immagine
        people[img] = bbox.loc[bbox['name'] == 'person']
    
    return crop(people)

def extract_people_in_videos(detections):
    '''
    Prende come input le detection fatte da 'video_detection()'. Poi per ogni frame determina le bounding box
    di tutte le persone nell'immagine. Applica infine crop che ritaglia dall'immagine la persona con il
    grado di confidenza più alto.

    '''
    # DETECTIONS: dizionario della forma - (video: {frame:crop_persona_con_piu_alta_confidenza})
    people = {}

    for video in detections:
        people[video] = {}
        for frame in detections[video].keys():
            # predictions (pandas)
            bbox = detections[video][frame].pandas().xyxy[0]  # xyxy=diagonale
            
            # vengono considerate solo le persone all'interno dell'immagine
            people[video][frame] = bbox.loc[bbox['name'] == 'person']

        # qua bisogna capire come fare
        video_crop(people)

def crop(people):
    '''
    @people: dizionario contentente il bbox delle persone all'interno dell'immagine. Viene presa quella con 
            più alta confidenza

    crop delle persone all'interno delle immagini del dataset
    
    @return: dizionario della forma {immagine: (x_top_left, y_top_left)}
    '''
    abs_path_dataset = os.path.abspath(DS_PATH)
    top_left_coords = {} # risultato da ritornare
    for img in people.keys():
        # VIENE CONSIDERATA UNA SOLA PERSONA (PER ADESSO). QUELLA CHE HA LA CONFIDENZA PIU' ALTA
        try:
            people[img] = people[img].loc[people[img]['confidence'] == max(people[img]['confidence'])]
            #people[img] = people[img].loc[people[img]['confidence'] > THRESHOLD]
            
            image_obj = Image.open(os.path.join(abs_path_dataset, img))
            cropped_image = image_obj.crop((people[img].iloc[0]['xmin'],
                                            people[img].iloc[0]['ymin'],
                                            people[img].iloc[0]['xmax'],
                                            people[img].iloc[0]['ymax']))
            
            # al nome del file non viene aggiunto nessun tag per avere lo stesso nome 
            # nella fase di analisi

            cropped_image.save(os.path.join(abs_path_dataset, "CROP", img))
            
            top_left_coords[img] = (people[img].iloc[0]['xmin'], people[img].iloc[0]['ymin'])

        except:
            pass
    return top_left_coords

def video_crop(people):
    abs_path_dataset = os.path.abspath(DS_PATH)
    for video in people:
        vidcap = cv2.VideoCapture(os.path.join(abs_path_dataset, video))
        success, image = vidcap.read()
        count = 0
        while success:
            try:
                # VIENE CONSIDERATA UNA SOLA PERSONA (PER ADESSO). QUELLA CHE HA LA CONFIDENZA PIU' ALTA
                people[video][count] = people[video][count].loc[people[video][count]['confidence'] == max(people[video][count]['confidence'])]

                imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_obj = Image.fromarray(imageRGB)
                cropped_image = image_obj.crop((people[video][count].iloc[0]['xmin'],
                                                people[video][count].iloc[0]['ymin'],
                                                people[video][count].iloc[0]['xmax'],
                                                people[video][count].iloc[0]['ymax']))
                #cropped_image.save(os.path.join(abs_path_dataset, "CROP", f"{video}_cropped_frame{count}.png"))
                cropped_image.save(os.path.join(abs_path_dataset, "CROP", f"{CROPPED_IMAGES_TAG}frame{count}{video.split('.')[0]}.png"))
            except:
                pass
            
            success, image = vidcap.read()
            
            count += 1

def normalize_ground_truth(gt, top_left_coords):
    '''
    @gt: dizionario della forma {immagine: {parte_corpo: (x_pc, y_pc), ...} }
    @top_left_coords: dizionario della forma {immagine: (x_top_left, y_top_left)}

    Normalizzo le coordinate del ground truth considerando adesso che l'immagine è cropped.
    
    @return: dizionario della forma {immagine: {parte: (x,y)}} con (x,y) normalizzato all'immagine cropped
    '''
    for img in top_left_coords:
        # per ogni body-part
        for bp in gt[img]:
            # l'head_box non è necessario cambiarlo poiché l'unica cosa che interessa successivamente è la diagonale
            # del suo bbox
            if bp == 'head_box':
                continue
            
            alpha, beta = gt[img][bp]
            x, y = top_left_coords[img]
            # alpha - x, beta - y
            gt[img][bp] = (alpha - x, beta - y)
    return gt
            
# COSTANTI
EFFICIENTPOSE_PATH = '../EfficientPose-master'
EFFICIENTPOSE_MAIN = 'track.py'
MODELS = ['RT','I','II','III','IV']
CROPPED_IMAGES_TAG = 'cropped_'
TRACKED_IMAGES_TAG = '_tracked'
JOINT_ID = {0:"right_ankle", 1:"right_knee", 2: "right_hip", 3: "left_hip", 4: "left_knee", 5: "left_ankle", 
            6: "pelvis", 7:"thorax", 8:"upper_neck", 9:"head_top", 10:"right_wrist", 11:"right_elbow", 
            12:"right_shoulder", 13:"left_shoulder", 14:"left_elbow", 15: "left_wrist"}

# VARIABILI
FRAMEWORK = 'tflite' #'pytorch'

MODEL = MODELS[0]
ANALYZE_ONE_MODEL = True  # True lavora su un solo modello (MODEL), False su tutti i modelli
# variabile usata quando ANALYZE_ONE_MODEL=True. Questa variabile dice se devono essere creati 
# o meno i csv
CREATE_CSV_INFERENCE = False
# variabile usata quando ANALYZE_ONE_MODEL=False. Questa variabile dice quale metrica impiegare
# per la comparazione dei modelli
METRIC_NAME = 'pckh'
TAU_FOR_PRETTIFY = 0.5

DS_PATH = './DATASET_quantitavio/'
PARTIAL_RES_PATH = './DATASET_quantitavio/CROP'
RES_PATH = './results/'
THRESHOLD = 0.5

VIDEO_EXTENTION = '.mp' #mp3/4

if __name__ == "__main__":
    # Eseguo il crop delle immagini e le salvo in PARTIAL_RES_PATH
    imgs_det, _ = detect() # i dataset contengono solo immagini
    top_left_coords = extract_people_in_images(imgs_det)
    
    # dopo il crop va fatta l'inferenza

    #           MAIN DI test_MPII.py (leggermente adattato per gestire le immagini cropped
    # ####################################################################################
    # ####################################################################################

    with open('annotations.pickle', 'rb') as fin:
        annotations = pickle.load(fin)
        
    # path assoluto della immagini cropped
    abs_partial_path = os.path.abspath(PARTIAL_RES_PATH)

    # vengono creati i csv se necessario
    if ANALYZE_ONE_MODEL:
        if CREATE_CSV_INFERENCE:
            create_csv_model_inference(abs_partial_path, MODEL)
            move_csv_in_model_dir(abs_partial_path, MODEL)
    
        # descrizione della struttura link: http://human-pose.mpi-inf.mpg.de/#download

        ground_truth = get_rows_from_annotations(annotations, abs_partial_path)
        
        # dato che il ground truth è riferito all'immagine completa e non al crop allora sistemo i punti
        ground_truth = normalize_ground_truth(ground_truth, top_left_coords)

        inference = get_inference(ground_truth.keys(), abs_partial_path, MODEL)    
        
        # [0] per il recupero del solo valore della metrica
        print("PCK:", utils_detector.pckh(ground_truth, inference)[0])
        prettify(utils_detector.pckh(ground_truth, inference, TAU_FOR_PRETTIFY)[1], MODEL, "pckh")
        print("PCP:", utils_detector.pcp(ground_truth, inference)[0])
        prettify(utils_detector.pcp(ground_truth, inference, TAU_FOR_PRETTIFY)[1], MODEL, "pcp")
        print("PDJ:", utils_detector.pdj(ground_truth, inference)[0])
        prettify(utils_detector.pdj(ground_truth, inference, TAU_FOR_PRETTIFY)[1], MODEL, "pdj")

        print("AUC per PCKH:", utils_detector.auc(utils_detector.pckh, ground_truth, inference, _max=1, visualize=True, model_name=MODEL))
        print("AUC per PCP:", utils_detector.auc(utils_detector.pcp, ground_truth, inference, _max=1, visualize=True, model_name=MODEL))
        print("AUC per PDJ:", utils_detector.auc(utils_detector.pdj, ground_truth, inference, _max=1, visualize=True, model_name=MODEL))
    
    else:
        if CREATE_CSV_INFERENCE:
            for m in MODELS:
                create_csv_model_inference(abs_partial_path, m)
                move_csv_in_model_dir(abs_partial_path, m)

        compararison = compare_models(annotations, abs_partial_path, METRIC_NAME, TAU_FOR_PRETTIFY)

        for m in MODELS:
            utils_detector.auc_sup(compararison[m][0],compararison[m][1], m, METRIC_NAME)