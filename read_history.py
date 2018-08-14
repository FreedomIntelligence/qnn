import os
import codecs
import numpy as np
from params import Params
path = 'case study/'
performance_dict = {}
params = Params()

from sklearn.neighbors import KDTree



import os
import codecs
import numpy as np
from scipy.spatial.distance import cosine
path = 'eval'
performance_dict = {}
params = Params()
from tqdm import tqdm

def complex_metric(number1, number2):    
    return np.linalg.norm(number1 * number2)


def write_to_file(filename,strings):
    with codecs.open(filename,"w",encoding="utf-8") as f:
        f.write("\n".join(strings))
        
def case_study(eval_dir):
    strings = []
    history = np.load(os.path.join(eval_dir,'history.npy'))
    phase = np.load(os.path.join(eval_dir,'amplitude_embedding.npy'))
    amplitude = np.load(os.path.join(eval_dir,'phase_embedding.npy'))
    weights = np.load(os.path.join(eval_dir,'weights.npy'))[:,0]
    measurements = np.load(os.path.join(eval_dir,'measurements.npy'))
    id2word = np.load(os.path.join(eval_dir,'id2word.npy'))
    
    config_path = os.path.join(eval_dir,'config.ini')
    # print(config_path)
    params.parse_config(config_path)
    params.export_to_config("sb.ini")
    
    strings.append(" ".join(id2word[np.argsort(weights[1:])[:50]]))
    
    strings.append(" ".join(id2word[np.argsort(weights[1:])[-50:]]))
    strings.append("\n")
    
    embedding = np.cos(phase)*amplitude+1j*np.sin(phase)*amplitude
    measuremment_vector = measurements[:,:,0] + 1j *measurements[:,:,1]
  
   
    for i in range(params.measurement_size):
        numbers=[]
        for j,word in tqdm(enumerate(embedding)):
            vector = measuremment_vector[i,:]
            sim =complex_metric(word, vector )
            numbers.append(sim)
        strings.append(" ".join(id2word[np.argsort(numbers[1:])[:50]]))
#            q.put(Job(distance,id2word[j+1]))
#            tree.query_ball_point(vector,1)
#            tree = KDTree(embedding,metric='pyfunc',func=complex_metric)
    write_to_file(params.dataset_name+".cases", strings)



for file_name in os.listdir(path):
    eval_dir = os.path.join(path, file_name)
    history_path = os.path.join(eval_dir,'history.npy')
    config_path = os.path.join(eval_dir,'config.ini')
    # print(config_path)
    params.parse_config(config_path)

    dataset_name = params.dataset_name
    # print(params.dataset_name)
    history = np.load(history_path).tolist()
    accuracy = max(history['val_acc'])
    if not dataset_name in performance_dict:
        performance_dict[dataset_name] = (accuracy, file_name)
    else:
        if accuracy > performance_dict[dataset_name][0]:
            performance_dict[dataset_name] = (accuracy, file_name)

for dataset, args in performance_dict.items():
    percision, filename = args
    
    eval_dir="eval\\" + filename
    case_study(eval_dir)













   
#def complex_metric(number1, number2):
#    
#    return np.linalg.norm(number1 * number2)



    
