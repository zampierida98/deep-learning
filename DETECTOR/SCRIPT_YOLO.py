# -*- coding: utf-8 -*-
'''
INSERIRE IL SEGUENTE CODICE NELLA ROOT DI: https://github.com/ultralytics/yolov5
INSERIRE LE IMMAGINI NELLA DIRECTORY: DS_PATH
'''

# IMPORT
from PIL import Image
import torch
import torchvision
import os
import warnings
warnings.simplefilter("ignore", FutureWarning)


# FUNZIONI
def model_inference(abs_ds_path, abs_res_path, model):
    # lista con i nomi delle immagini
    list_of_file = []
    for file in os.listdir(abs_ds_path):
        # controllo se è un file e se all'interno del nome è presente CROPPED_IMAGES_TAG ('cropped_') segno
        # che l'immagine è stata ritagliata
        if os.path.isfile(os.path.join(abs_ds_path, file)) and CROPPED_IMAGES_TAG in file:
            list_of_file.append(f'{abs_ds_path}/{file}')
    
    # se il processo non si trova già in EFFICIENTPOSE_PATH viene fatto il cambio di directory
    try:
        os.chdir(EFFICIENTPOSE_PATH)
    except:
        pass

    for file in list_of_file:
        os.system(f'python {EFFICIENTPOSE_MAIN} --path="{file}" --framework={FRAMEWORK} --model={model} {OUTPUT}')

    # sposto i risultati in RES_PATH
    for file in os.listdir(abs_ds_path):
        if os.path.isfile(os.path.join(abs_ds_path,file)) and TRACKED_IMAGES_TAG in file:
            os.replace(f'{abs_ds_path}/{file}', f'{abs_res_path}/{model}/{file}')

def infer():
    # path assoluto della cartella DS_PATH
    abs_ds_path = os.path.abspath(DS_PATH)
    abs_res_path = os.path.abspath(RES_PATH)
    
    # vengono prodotti i risultati
    if ANALYZE_ONE_MODEL:
        model_inference(abs_ds_path, abs_res_path, MODEL)

    else:
        for m in MODELS:
            model_inference(abs_ds_path, abs_res_path, m)
    
    # rimuovo i file cropped
    for file in os.listdir(abs_ds_path):
        if os.path.isfile(os.path.join(abs_ds_path, file)) and CROPPED_IMAGES_TAG in file:
            os.remove(f'{abs_ds_path}/{file}')

def video_detection(video, model):
    res = {}
    for ind, frame in enumerate(video):
        res[ind] = model(frame['data']).print()
    
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
            video_detections[img] = video_detection(torchvision.io.VideoReader(os.path.join(abs_path_dataset, img)), model)

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
    
    crop(people)

def crop(people):
    for img in people.keys():
        # VIENE CONSIDERATA UNA SOLA PERSONA (PER ADESSO). QUELLA CHE HA LA CONFIDENZA PIU' ALTA
        people[img] = people[img].loc[people[img]['confidence'] == max(people[img]['confidence'])]
        #people[img] = people[img].loc[people[img]['confidence'] > THRESHOLD]
        
        image_obj = Image.open(os.path.join(os.path.abspath(DS_PATH), img))
        cropped_image = image_obj.crop((people[img].iloc[0]['xmin'],
                                        people[img].iloc[0]['ymin'],
                                        people[img].iloc[0]['xmax'],
                                        people[img].iloc[0]['ymax']))
        
        cropped_image.save(os.path.join(os.path.abspath(DS_PATH), CROPPED_IMAGES_TAG+img))

# COSTANTI
EFFICIENTPOSE_PATH = '../EfficientPose-master'
EFFICIENTPOSE_MAIN = 'track.py'
MODELS = ['RT','I','II','III','IV']
CROPPED_IMAGES_TAG = 'cropped_'
TRACKED_IMAGES_TAG = '_tracked'

# VARIABILI
FRAMEWORK = 'tflite' #'pytorch'
OUTPUT = '--visualize'

MODEL = MODELS[0]
ANALYZE_ONE_MODEL = True  # True lavora su un solo modello (MODEL), False su tutti i modelli

DS_PATH = './DATASET/'
RES_PATH = './results/'
THRESHOLD = 0.5

VIDEO_EXTENTION = '.mp' #mp3/4

if __name__ == "__main__":
    imgs_det, videos_det = detect()
    extract_people_in_images(imgs_det)
    #extract_people_in_videos(videos_det)

    # dopo il crop va fatta l'inferenza
    infer()