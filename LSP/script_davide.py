# -*- coding: utf-8 -*-
# IMPORT
import os
import warnings
warnings.simplefilter("ignore", FutureWarning)

# COSTANTI
DS_PATH = './batch/'
EFFICIENTPOSE_PATH = './EfficientPose-master/'
EFFICIENTPOSE_MAIN = 'track.py'
MODELS = ['RT','I','II','III','IV']
FRAMEWORK = 'pytorch'
MODEL = MODELS[0]

# VARIABILI
# True lavora su un solo modello (MODEL), False crea un plot che compara i diversi modelli
ANALYZE_ONE_MODEL = False

# FUNZIONI
def create_csv_model_inference(abs_ds_path, model):
    # lista con i nomi delle sotto-cartelle
    dirs = [ f for f in os.listdir(abs_ds_path) if not os.path.isfile(os.path.join(abs_ds_path,f)) ]

    for d in dirs:
        dir_path = os.path.join(abs_ds_path,d)
        
        # lista con i nomi delle immagini
        list_of_file = []
        for file in os.listdir(dir_path):
            if os.path.isfile(os.path.join(dir_path,file)) and not file.endswith(".csv"):
                list_of_file.append(f'{dir_path}/{file}')
        
        # se il processo non si trova già in EFFICIENTPOSE_PATH viene fatto il cambio di directory
        try:
            os.chdir(EFFICIENTPOSE_PATH)
        except:
            pass

        # per ogni file viene creato il csv
        for file in list_of_file:
            os.system(f'python3.7 {EFFICIENTPOSE_MAIN} --model={model} --path="{file}" --store --framework={FRAMEWORK}')

        # sposto i csv nella cartella dei risultati (la creo se non esiste)
        try:
            os.mkdir(os.path.abspath(f'./results/{model}'))
        except:
            pass

        for file in os.listdir(dir_path):
            if os.path.isfile(os.path.join(dir_path,file)) and file.endswith(".csv"):
                os.replace(f'{dir_path}/{file}', os.path.abspath(f'./results/{model}/{file}'))

def main_LSP():
    # path assoluto della cartella DS_PATH
    abs_ds_path = os.path.abspath(DS_PATH)
    
    # vengono creati i csv
    if ANALYZE_ONE_MODEL:
        create_csv_model_inference(abs_ds_path, MODEL)

    else:
        for m in MODELS:
            create_csv_model_inference(abs_ds_path, m)


if __name__ == "__main__":
    main_LSP()
