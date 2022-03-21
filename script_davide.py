# -*- coding: utf-8 -*-
'''
Il programma prende i modelli messi a disposizione da EfficientPose per testarne la bontà sul dataset MPII.
L'insieme delle funzioni di supporto (ad es. metriche impiegate) viene scritto in un file ausiliario così da poterle usare in un momento successivo
'''

# IMPORT
import os
import warnings
warnings.simplefilter("ignore", FutureWarning)

# COSTANTI
DS_PATH = './LSP/batch/3'
EFFICIENTPOSE_PATH = './EfficientPose-master/'
EFFICIENTPOSE_MAIN = 'track.py'
MODELS = ['RT','I','II','III','IV']
FRAMEWORK = 'pytorch'
MODEL = MODELS[0]

# VARIABILI
# True lavora su un solo modello (MODEL), False crea un plot che compara i diversi modelli
one_or_more = False
# variabile che dice se devono essere creati o meno i csv
create_csv_model_infer = True
# variabile che dice quale metrica impiegare per la comparazione dei modelli
metric_name = 'pckh'

# FUNZIONI
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

def move_csv_in_model_dir(abs_ds_path, model):
    # errore se la dir esiste già
    try:
        os.mkdir(f'{abs_ds_path}/{model}')
    except:
        pass
    
    for file in os.listdir(abs_ds_path):
        if file.endswith(".csv"):
            os.replace(f'{abs_ds_path}/{file}', f'{abs_ds_path}/{model}/{file}')

def main_LSP():
    # path assoluto della dir DS_PATH
    abs_ds_path = os.path.abspath(DS_PATH)
    
    # lista con i nomi delle immagini per get_inference
    images = [ f for f in os.listdir(abs_ds_path) if os.path.isfile(os.path.join(abs_ds_path,f)) ]

    # vengono creati i csv se necessario
    if one_or_more:
        if create_csv_model_infer:
            create_csv_model_inference(abs_ds_path, MODEL)
            move_csv_in_model_dir(abs_ds_path, MODEL)

    else:
        if create_csv_model_infer:
            for m in MODELS:
                create_csv_model_inference(abs_ds_path, m)
                move_csv_in_model_dir(abs_ds_path, m)


if __name__ == "__main__":
    main_LSP()
