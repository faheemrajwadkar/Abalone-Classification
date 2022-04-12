import pandas as pd
import numpy as np
import pickle

# add input cleaning funciton here
def sex_mapper(sex):
    if sex == 'Male':
        return 0, 1
    elif sex == 'Infant':
        return 1, 0
    else:
        return 0, 0
    
def create_features(height, shucked_weight, shell_weight, sex, model):
    sex_I, sex_M = sex_mapper(sex)
    input_features = pd.DataFrame(np.array([1.0, height, shucked_weight, shell_weight, sex_I, sex_M]).reshape(1, -1), 
                                  columns=model.feature_names_in_)
    return input_features

def prediction_mapper(prediction):
    if prediction == 0:
        return "The age of Abalone is 10 or less"
    else:
        return "The age of Abalone is over 10"
