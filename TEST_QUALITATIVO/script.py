'''
Il programma prende i modelli messi a disposizione da EfficientPose-master per testare la bontà dei
modelli sul video prescelto'''
# IMPORT
import pickle
import os
import utils
import numpy as np
import warnings
from itertools import groupby
from PIL import Image
warnings.simplefilter("ignore", FutureWarning)

# FUNZIONI
def move_result_in_model_dir(abs_ds_path, model, filename):
    # errore se la dir esiste già
    try:
        os.mkdir(f'{abs_ds_path}\\{model}')
    except:
        pass
    os.replace(f'{abs_ds_path}\\{filename}', f'{abs_ds_path}\\{model}\\{filename}')


# COSTANTI
DS_PATH = './dataset/'
EFFICIENTPOSE_PATH = '../EfficientPose-master/'
EFFICIENTPOSE_MAIN = 'track.py'
MODELs = ['RT','I','II', 'III', 'IV']

# VARIABILI
FRAMEWORK = 'tflite' #, 'keras' #"torch"# "tf", #
MODEL = MODELs[0]
# True lavora su un solo modello (MODEL), False crea un plot che compara i diversi modelli
ANALYZE_ONE_MODEL = True

VIDEO_FILENAME = '/gveii.mp4'
VIDEO_ANALYSIS_FILENAME = 'gveii_tracked.mp4'


# MAIN
if __name__ == "__main__":
    abs_ds_path = os.path.abspath(DS_PATH)
    file = abs_ds_path + VIDEO_FILENAME
    # se il processo non si trova già in EFFICIENTPOSE_PATH viene fatto il cambio di directory
    try:
        os.chdir(EFFICIENTPOSE_PATH)
    except:
        pass

    if ANALYZE_ONE_MODEL:
        os.system(f'python {EFFICIENTPOSE_MAIN} --model={MODEL} --path="{file}" --visualize --framework={FRAMEWORK}')
    else:
        for model in MODELs:
            os.system(f'python {EFFICIENTPOSE_MAIN} --model={model} --path="{file}" --visualize --framework={FRAMEWORK}')
            move_result_in_model_dir(abs_ds_path, model, VIDEO_ANALYSIS_FILENAME)
