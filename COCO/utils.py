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
    Distanza euclidea tra due punti bi-dimensionali
    '''
    return ((x2-x1)**2 + (y2-y1)**2)**(1/2)

def get_segments_16_parts():
    '''
    Ritorna la lista di coppie dove ciascune di esse rappresenta una parte di un arto.
    '''
    return [('head_top', 'upper_neck'), ('upper_neck', 'thorax'), ('thorax', 'right_shoulder'), ('thorax', 'left_shoulder'), 
                ('thorax', 'pelvis'), ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'), ('left_shoulder', 'left_elbow'),
                ('left_elbow', 'left_wrist'), ('pelvis', 'right_hip'), ('pelvis', 'left_hip'), ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'), ('left_hip', 'left_knee'), ('left_knee', 'left_ankle')]

def get_16_parts():
    return ["right_ankle", "right_knee", "right_hip", "left_hip",
            "left_knee", "left_ankle", "pelvis", "thorax",
            "upper_neck", "head_top", "right_wrist", "right_elbow", 
            "right_shoulder", "left_shoulder", "left_elbow", "left_wrist"]

def pck(ground_truth, inference, tau=0.5):
    '''
    PCK basato sulla diagonale del torso
    '''
    res = 0
    spec_res = {p:0 for p in get_16_parts()}
    for img in ground_truth:
        d = max(distance(ground_truth[img]['right_shoulder'][0], ground_truth[img]['right_shoulder'][1], 
                                  ground_truth[img]['left_hip'][0], ground_truth[img]['left_hip'][1]), 
                        distance(ground_truth[img]['left_shoulder'][0], ground_truth[img]['left_shoulder'][1], 
                                  ground_truth[img]['right_hip'][0], ground_truth[img]['right_hip'][1]))
        correct_pred = 0
        for p in inference[img]:
            # ci sono immagini in cui lo scheletro potrebbe essere parzialmente osservabile
            try:
                x1, y1 = inference[img][p]
                x2, y2 = ground_truth[img][p]
                if distance(x1, y1, x2, y2) <= tau*d:
                    correct_pred += 1
                    spec_res[p] += 1
            except:
                pass
        res += correct_pred / (len(ground_truth[img]))
    return res / len(ground_truth), {k:(spec_res[k] / len(ground_truth)) for k in spec_res}

def pcp(ground_truth, inference, tau=0.5):
    '''
    Percentage of Correct Parts (PCP)
    A limb is considered detected and a correct part if the distance between the
    two predicted joint locations and the true limb joint locations is at most half
    of the limb length (PCP tau=0.5).

    Measures detection rate of limbs
    '''
    segments = get_segments_16_parts()
    res = 0
    spec_res = {(bp1, bp2):0 for bp1, bp2 in segments}

    for img in ground_truth:        
        counter = 0
        correct_pred = 0
        for bp1, bp2 in segments:
            # ci sono immagini in cui lo scheletro potrebbe essere parzialmente osservabile
            try:
                bp1x1, bp1y1 = inference[img][bp1]
                bp2x1, bp2y1 = inference[img][bp2]
                bp1x2, bp1y2 = ground_truth[img][bp1]
                bp2x2, bp2y2 = ground_truth[img][bp2]
                if (distance(bp1x1, bp1y1, bp1x2, bp1y2) <= tau * distance(bp1x2, bp1y2, bp2x2, bp2y2) and 
                    distance(bp2x1, bp2y1, bp2x2, bp2y2) <= tau * distance(bp1x2, bp1y2, bp2x2, bp2y2)):
                    correct_pred += 1
                    spec_res[(bp1, bp2)] += 1

                counter += 1
            except:
                pass
        res += correct_pred / counter
    return res / len(ground_truth), {k:(spec_res[k] / len(ground_truth)) for k in spec_res}

def pdj(ground_truth, inference, tau=0.5):
    '''
    Percentage of Detected Joints (PDJ)
    In order to fix the issue raised by PCP, a new metric was proposed.
    It measures the distance between the predicted and the true joint within a certain fraction 
    of the torso diameter and it is called the percentage of detected joints (PDJ).  
    PDJ helps to achieve localization precision, which alleviates the drawback of PCP since the
    detection criteria for all joints are based on the same distance threshold.
    '''

    segments = get_segments_16_parts()
    res = 0
    spec_res = {(bp1, bp2):0 for bp1, bp2 in segments}
    
    for img in ground_truth:        
        counter = 0
        correct_pred = 0
        torso_diag = max(distance(ground_truth[img]['right_shoulder'][0], ground_truth[img]['right_shoulder'][1], 
                                  ground_truth[img]['left_hip'][0], ground_truth[img]['left_hip'][1]), 
                        distance(ground_truth[img]['left_shoulder'][0], ground_truth[img]['left_shoulder'][1], 
                                  ground_truth[img]['right_hip'][0], ground_truth[img]['right_hip'][1]))
                                
        for bp1, bp2 in segments:
            # ci sono immagini in cui lo scheletro potrebbe essere parzialmente osservabile
            try:
                bp1x1, bp1y1 = inference[img][bp1]
                bp2x1, bp2y1 = inference[img][bp2]
                bp1x2, bp1y2 = ground_truth[img][bp1]
                bp2x2, bp2y2 = ground_truth[img][bp2]

                if (distance(bp1x1, bp1y1, bp1x2, bp1y2) <= tau * torso_diag and 
                    distance(bp2x1, bp2y1, bp2x2, bp2y2) <= tau * torso_diag):
                    correct_pred += 1
                    spec_res[(bp1, bp2)] += 1

                counter += 1
            except:
                pass
        res += correct_pred / counter
    return res / len(ground_truth), {k:(spec_res[k] / len(ground_truth)) for k in spec_res}

def auc_sup(X,Y, model_name, metric_name, visualize=True):
    AUC = round(integrate.trapz(Y, X) / (X[-1] - X[0]), 4)
    if visualize:
        plot({model_name: (X,Y)}, f"auc per {metric_name} = {AUC}")
    return AUC

def auc(metric, ground_truth, inference, _min=0, _max=0.5, step=0.01, visualize=False, model_name=''):
    X = [i for i in np.arange(_min, _max, step)]
    Y = [metric(ground_truth, inference, x) for x in X]
    # recupero solo il valore della metrica
    Y = [r1 for (r1,_) in Y]
    return auc_sup(X,Y, model_name, metric.__name__, visualize)


def plot(dict_values, metric_name):
    '''
    dict_values: è un dizionario la cui chiave è il nome del modello mentre il valore è la coppia (X,Y)
    metric_name: nome della metrica usata per generare i numeri dentro Y
    '''
    _, ax = plt.subplots()
    for model in dict_values:
        ax.plot(dict_values[model][0], dict_values[model][1])
    ax.legend(['EfficientPose '+model for model in dict_values])
    plt.xticks(rotation=90)
    plt.title(f"Metrica {metric_name}".upper())
    plt.show()


if __name__ == "__main__":
    pass