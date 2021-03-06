import numpy as np
from scipy import integrate
from scipy.io import loadmat, matlab
import matplotlib.pyplot as plt

# FUNZIONI
def load_mat(filename):
    '''
    Il metodo scipy.io.loadmat non genera una struttura dati basata su liste o dizionari.
    Per cui ricorsivamente si guarda il tipo di struttura ottenuta da scipy.io.loadmat
    e la si converte nelle strutture python corrispondenti partendo dal basso fino ad arrivare in cima.
    '''

    def inner_converter(obj):
        '''
        Esegue la chiamata alle funzioni ricorsive che creano obj python
        '''
        for k in obj:
            if type(obj[k]) == matlab.mio5_params.mat_struct:
                obj[k] = inner_todict(obj[k])
            elif type(obj[k]) == np.ndarray:
                obj[k] = inner_toarray(obj[k])
        return obj

    def inner_todict(obj):
        '''
        inner_todict ricorsivamente trasforma i valori in oggetti python usando
        un approccio simile alla funzione inner_converter
        '''
        res = {}
        for k in obj._fieldnames: # per ogni campo
            value = obj.__dict__[k] # prendiamo il valore del campo k
            if type(value) == matlab.mio5_params.mat_struct:
                res[k] = inner_todict(value)
            elif type(value) == np.ndarray:
                res[k] = inner_toarray(value)
            else:
                res[k] = value
        return res

    def inner_toarray(obj):
        '''
        Gli np.ndarray possono non essere semplici tensori su numeri
        ma anche cellarrays (una sorta di liste al cui interno troviamo vari tipi di oggetti).
        Per cui se è un cellarray (ndarray.dtype != 'float64') allora va applicato
        la procedura nota di conversione.
        '''
        if obj.dtype != 'float64':
            res = []
            for el in obj:
                if type(el) == matlab.mio5_params.mat_struct:
                    res.append(inner_todict(el))
                elif type(el) == np.ndarray:
                    res.append(inner_toarray(el))
                else:
                    res.append(el)
            return np.array(res)
        else:
            return obj

    obj = loadmat(filename, struct_as_record=False, squeeze_me=True)
    return inner_converter(obj)

def distance(x1, y1, x2, y2):
    '''
    Distanza euclidea tra due punti bi-dimensionali.
    '''
    return ((x2-x1)**2 + (y2-y1)**2)**(1/2)

def get_segments():
    '''
    Ritorna una lista di coppie dove ciascuna di esse rappresenta un'articolazione.
    '''
    return [('head_top', 'neck'), ('neck', 'right_shoulder'), ('neck', 'left_shoulder'), 
            ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
            ('neck', 'pelvis'), ('pelvis', 'right_hip'), ('pelvis', 'left_hip'),
            ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
            ('left_hip', 'left_knee'), ('left_knee', 'left_ankle')]

def get_torso(gt_img):
    bp1 = 'neck'
    bp2 = 'pelvis'
    bp1x, bp1y = gt_img[bp1]
    bp2x, bp2y = gt_img[bp2]
    return distance(bp1x, bp1y, bp2x, bp2y)

    # return max(distance(gt_img['right_shoulder'][0], gt_img['right_shoulder'][1], 
    #                     gt_img['left_hip'][0], gt_img['left_hip'][1]), 
    #            distance(gt_img['left_shoulder'][0], gt_img['left_shoulder'][1], 
    #                     gt_img['right_hip'][0], gt_img['right_hip'][1]))

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
        
        #print(bp1, bp2, correct_pred, counter)
        # se il segmento non esiste non lo conto
        try:
            res[(bp1, bp2)] = round(correct_pred / counter, 3) * 100
            res['total'] += correct_pred / counter
        except:
            pass

    res['total'] = round(res['total'] / (len(res.keys())-1), 3) * 100
    
    return res

def pdj(ground_truth, inference, tau=0.5):
    segments = get_segments()
    res = {'total': 0}
    for bp1, bp2 in segments:
        counter = 0  # ci sono immagini in cui lo scheletro potrebbe essere parzialmente osservabile
        correct_pred = 0
        
        for img in ground_truth:
            try:
                torso_diag = get_torso(ground_truth[img])

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
        
        #print(bp1, bp2, correct_pred, counter)
        # se il segmento non esiste non lo conto
        try:
            res[(bp1, bp2)] = round(correct_pred / counter, 3) * 100
            res['total'] += correct_pred / counter
        except:
            pass
    
    res['total'] = round(res['total'] / (len(res.keys())-1), 3) * 100
    return res

def pck(ground_truth, inference, tau=0.5):
    JOINT_ID_LSP = {0:"right_ankle", 1:"right_knee", 2:"right_hip", 3:"left_hip",
                    4:"left_knee", 5:"left_ankle", 6:"right_wrist", 7:"right_elbow",
                    8:"right_shoulder", 9:"left_shoulder", 10:"left_elbow",
                    11:"left_wrist", 12:"neck", 13:"head_top"}
    
    res = {'total': 0}
    for k, p in JOINT_ID_LSP.items():
        counter = 0  # ci sono immagini in cui lo scheletro potrebbe essere parzialmente osservabile
        correct_pred = 0
        
        for img in ground_truth:
            try:
                torso_diag = get_torso(ground_truth[img])

                x1, y1 = inference[img][p]
                x2, y2 = ground_truth[img][p]
                if distance(x1, y1, x2, y2) <= tau * torso_diag:
                    correct_pred += 1
                counter += 1
            except:
                pass
        
        #print(p, correct_pred, counter)
        # se il keypoint non esiste non lo conto
        try:
            res[p] = round(correct_pred / counter, 3) * 100
            res['total'] += correct_pred / counter
        except:
            pass
        
    res['total'] = round(res['total'] / (len(res.keys())-1), 3) * 100
    return res

def auc(metric, ground_truth, inference, _min=0, _max=1, step=0.01):
    X = [i for i in np.arange(_min, _max, step)]
    Y = [metric(ground_truth, inference, x)['total']/100 for x in X]
    AUC = integrate.trapz(Y,X)
    AUC = round(AUC / (_max - _min), 3)  # normalizzazione
    return (X,Y,AUC)

def plot(metric, values):
    _, ax = plt.subplots(figsize=(20, 10), dpi=60)
    for m in values:
        ax.plot(values[m][0], values[m][1])

    ax.legend(['EfficientPose '+m for m in values])
    plt.xticks(rotation=90)
    plt.title(f'comparazione dei valori per la metrica {metric.__name__}'.upper())
    plt.ylabel(metric.__name__)
    plt.xlabel('soglia')
    plt.show()
