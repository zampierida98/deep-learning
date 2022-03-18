from ssl import ALERT_DESCRIPTION_INSUFFICIENT_SECURITY
import numpy as np
import pickle
from scipy import integrate
from scipy.io import loadmat, matlab
import matplotlib.pyplot as plt

# FUNZIONI
def load_mat(filename):
    """
    This function should be called instead of direct scipy.io.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """

    def _check_vars(d):
        """
        Checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in d:
            if isinstance(d[key], matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
            elif isinstance(d[key], np.ndarray):
                d[key] = _toarray(d[key])
        return d

    def _todict(matobj):
        """
        A recursive function which constructs from matobjects nested dictionaries
        """
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _toarray(elem)
            else:
                d[strg] = elem
        return d

    def _toarray(ndarray):
        """
        A recursive function which constructs ndarray from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        """
        if ndarray.dtype != 'float64':
            elem_list = []
            for sub_elem in ndarray:
                if isinstance(sub_elem, matlab.mio5_params.mat_struct):
                    elem_list.append(_todict(sub_elem))
                elif isinstance(sub_elem, np.ndarray):
                    elem_list.append(_toarray(sub_elem))
                else:
                    elem_list.append(sub_elem)
            return np.array(elem_list)
        else:
            return ndarray

    data = loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_vars(data)

def distance(x1, y1, x2, y2):
    return ((x2-x1)**2 + (y2-y1)**2)**(1/2)

def get_segments_16_parts():
    return [('head_top', 'upper_neck'), ('upper_neck', 'thorax'), ('thorax', 'right_shoulder'), ('thorax', 'left_shoulder'), 
                ('thorax', 'pelvis'), ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'), ('left_shoulder', 'left_elbow'),
                ('left_elbow', 'left_wrist'), ('pelvis', 'right_hip'), ('pelvis', 'left_hip'), ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'), ('left_hip', 'left_knee'), ('left_knee', 'left_ankle')]

def pckh(ground_truth, inference, tau=0.5):
    res = 0
    for img in ground_truth:
        head_box = ground_truth[img]['head_box']
        d = distance(head_box[0],head_box[1],head_box[2],head_box[3])
        l = 0.6 * d

        correct_pred = 0
        for p in inference[img]:
            # ci sono immagini in cui lo scheletro potrebbe essere parzialmente osservabile
            try:
                x1, y1 = inference[img][p]
                x2, y2 = ground_truth[img][p]
                if distance(x1, y1, x2, y2) <= tau*l:
                    correct_pred += 1
            except:
                pass
        res += correct_pred / (len(ground_truth[img]) - 1) # il gt contiene anche il bounding box della testa
    return res / len(ground_truth)

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
                counter += 1
            except:
                pass
        res += correct_pred / counter
    return res / len(ground_truth)

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
    for img in ground_truth:        
        counter = 0
        correct_pred = 0
        try:
            torso_diag = max(distance(ground_truth[img]['right_shoulder'][0], ground_truth[img]['right_shoulder'][1], 
                                      ground_truth[img]['left_hip'][0], ground_truth[img]['left_hip'][1]), 
                            distance(ground_truth[img]['left_shoulder'][0], ground_truth[img]['left_shoulder'][1], 
                                      ground_truth[img]['right_hip'][0], ground_truth[img]['right_hip'][1]))
                                    
            for bp1, bp2 in segments:
                # ci sono immagini in cui lo scheletro potrebbe essere parzialmente osservabile
                bp1x1, bp1y1 = inference[img][bp1]
                bp2x1, bp2y1 = inference[img][bp2]
                bp1x2, bp1y2 = ground_truth[img][bp1]
                bp2x2, bp2y2 = ground_truth[img][bp2]

                if (distance(bp1x1, bp1y1, bp1x2, bp1y2) <= tau * torso_diag and 
                    distance(bp2x1, bp2y1, bp2x2, bp2y2) <= tau * torso_diag):

                    correct_pred += 1
                counter += 1
            
            res += correct_pred / counter
        except:
            pass
        
    return res / len(ground_truth)

def auc(metric, ground_truth, inference, _min=0, _max=0.5, step=0.01, visualize=False, model_name=''):
    X = [i for i in np.arange(_min, _max, step)]
    Y = [metric(ground_truth, inference, x) for x in X]
    AUC = round(integrate.trapz(X, Y), 4)

    if visualize:
        plot({model_name: (X,Y)}, "auc per " + metric.__name__)
    return AUC

def auc_2(X,Y, model_name, metric_name):
    AUC = round(integrate.trapz(X, Y), 4)
    plot({model_name: (X,Y)}, f"auc per {metric_name} = {AUC}")
    return AUC


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
    mat = load_mat('annotations.mat')
    
    train_test_imgs = mat['RELEASE']['img_train']  # 0=test, 1=train
    annotations = {}
    for i in range(len(train_test_imgs)):
        if train_test_imgs[i] == 1 and type(mat['RELEASE']['annolist'][i]['annorect']) == dict:
            annotations[mat['RELEASE']['annolist'][i]['image']['name']] = mat['RELEASE']['annolist'][i]

    with open("annotations.pickle", 'wb') as fout:  # Overwrites any existing file.
        pickle.dump(annotations, fout, pickle.HIGHEST_PROTOCOL)