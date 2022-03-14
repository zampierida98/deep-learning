import numpy as np
import pickle
from scipy.io import loadmat, matlab

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

def pckh(ground_truth, inference, tau=0.5):
    res = 0
    for img in ground_truth:
        head_box = ground_truth[img]['head_box']
        d = distance(head_box[0],head_box[1],head_box[2],head_box[3])
        l = 0.6 * d
        
        cont = 0
        for p in inference[img]:
            # ci sono immagini in cui lo scheletro potrebbe essere parzialmente osservabile
            try:
                x1, y1 = inference[img][p]
                x2, y2 = ground_truth[img][p]
                if distance(x1, y1, x2, y2) <= tau*l:
                    cont += 1
            except:
                pass
        res += cont / (len(ground_truth[img])-1)  # non conto head_box
    return res / len(ground_truth)

if __name__ == "__main__":
    mat = load_mat('annotations.mat')
    
    train_test_imgs = mat['RELEASE']['img_train']  # 0=test, 1=train
    annotations = {}
    for i in range(len(train_test_imgs)):
        if train_test_imgs[i] == 1 and type(mat['RELEASE']['annolist'][i]['annorect']) == dict:
            annotations[mat['RELEASE']['annolist'][i]['image']['name']] = mat['RELEASE']['annolist'][i]

    with open("annotations.pickle", 'wb') as fout:  # Overwrites any existing file.
        pickle.dump(annotations, fout, pickle.HIGHEST_PROTOCOL)