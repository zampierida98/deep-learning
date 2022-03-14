'''
Il programma prende i modelli messi a disposizione da EfficientPose-master per testare la bontà dei
modelli sul dataset MPII. L'insieme delle funzioni di supporto come la metrica impiegata viene scritto
in un file ausiliario così da poterla usare in un momento successivo
'''
# IMPORT
import os
import sys
# VARIABILI E COSTANTI

# dataset path
#DS_PATH = './MPII/images'
DS_PATH = './MPII/sub_dataset'

EFFICIENTPOSE_PATH = './EfficientPose-master/'
EFFICIENTPOSE_MAIN = 'track.py'
MODEL = ['RT', 'RT_Lite','I','I_Lite', 'II','II_Lite', 'III', 'IV'][5]
# FUNZIONI
# MAIN
if __name__ == "__main__":
    for file in os.listdir(DS_PATH):
        abs_file_path = os.path.abspath(DS_PATH) + "/"
        os.system(f'python {EFFICIENTPOSE_PATH+EFFICIENTPOSE_MAIN}.py -m={MODEL} -p={abs_file_path+file} -s')
