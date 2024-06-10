
import os
import pandas as pd
import numpy as np

def read_dataset(data_name):
    '''
    =====A demo dataset======
    '''
    if data_name == 'demo':
        data_dir = './dataset/demo/'
        class_name = ['normal.', 'neptune.', 'smurf.', 'back.']
        if not os.path.exists(data_dir + 'pro_data.csv'):
            p = pd.read_csv(data_dir + 'kddcup.data_10_percent', error_bad_lines=False, header=None)
            p_part = pd.DataFrame()
            for i1 in class_name:
                p_part = pd.concat([p_part, p[p[41]==i1]], axis = 0)
            for i2 in [1, 2, 3]:
                key_ls = list((p_part[i2].value_counts()).keys())
                for i3 in key_ls:
                    p_part[i2][p_part[i2]==i3] = key_ls.index(i3)
            for i4 in class_name:
                p_part[41][p_part[41]==i4] = class_name.index(i4)
            p_part.to_csv(data_dir + 'pro_data.csv', index=False)
        data = pd.read_csv(data_dir + 'pro_data.csv')
        return data.values[:,:-1], data.values[:,-1]
