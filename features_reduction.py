import numpy as np # підключення NUMPY
from dataprocessing_lib import storing_data as store
from dataprocessing_lib import load_data as load
import matplotlib.pyplot as plt
import pandas as pd
import dataprocessing_lib as dl
from sklearn.preprocessing import StandardScaler


# a = np.array([[1],
#               [2],
#               [3]])
# b = np.array([[4],[5],[6]])
# c = np.concatenate((a, b), axis=1)
# print(c)
# exit(0)




#---------------------завантажимо данні--------------------
file_dir = 'Features_calculated'
#норма
norm_features_unprepeared = load(file_dir,'NORMAL_features.npz')
#паталогія
echemia_features_unprepeared = load(file_dir,'ESHEMIA_features.npz')

# Створюємо пандас датафрейм
colums_names = ['(i) normalA', '(i) sttosp', '(i) sttossig', '(i) S_st_segm', '(ii) normalA',
         '(ii) sttosp', '(ii) sttossig', '(ii) S_st_segm', '(iii) normalA',
         '(iii) sttosp', '(iii) sttossig', '(iii) S_st_segm', '(avr) normalA',
         '(avr) sttosp', '(avr) sttossig', '(avr) S_st_segm', '(avl) normalA',
         '(avl) sttosp', '(avl) sttossig', '(avl) S_st_segm', '(avf) normalA',
         '(avf) sttosp', '(avf) sttossig', '(avf) S_st_segm']
print('filtering normal')
norm_features_filtered = dl.emission_remove(norm_features_unprepeared,colums_names)
print('filtering patalogy')
print(norm_features_filtered)

echemia_features_filtered = dl.emission_remove(echemia_features_unprepeared,colums_names,remove=1)


#x = dl.increase_features_num(norm_features_filtered,30000,plotflag=0)
echemia_features_filtered_reduced = dl.reduse_features_num(echemia_features_filtered,np.shape(norm_features_filtered)[0],plotflag = 0)

file_dir = 'Features_calculated/balanced features'
store(norm_features_filtered,file_dir,'norm_features_filtered')
store(echemia_features_filtered_reduced,file_dir,'echemia_features_filtered_reduced')
# scaler = StandardScaler()
# data = norm_features_filtered
# print(data)
# coefs = scaler.fit(data)
#
# scaled_data =coefs.transform(data)
# print(scaled_data)

