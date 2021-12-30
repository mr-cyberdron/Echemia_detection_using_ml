import numpy as np # підключення NUMPY
from dataprocessing_lib2 import storing_data as store
from dataprocessing_lib2 import load_data as load
import pandas as pd
import dataprocessing_lib2 as dl
from sklearn.preprocessing import StandardScaler
import pickle
from os.path import dirname, join as pjoin #os - функції для роботи з операційною системою,
import seaborn as sns
import matplotlib.pyplot as plt
colums_names = ['(i) normalA', '(i) sttosp', '(i) sttossig', '(i) S_st_segm', '(ii) normalA',
         '(ii) sttosp', '(ii) sttossig', '(ii) S_st_segm', '(iii) normalA',
         '(iii) sttosp', '(iii) sttossig', '(iii) S_st_segm', '(avr) normalA',
         '(avr) sttosp', '(avr) sttossig', '(avr) S_st_segm', '(avl) normalA',
         '(avl) sttosp', '(avl) sttossig', '(avl) S_st_segm', '(avf) normalA',
         '(avf) sttosp', '(avf) sttossig', '(avf) S_st_segm']

#завантажуэмо данні

file_dir = 'Features_calculated/balanced features'

normal_data=load(file_dir,'norm_features_filtered.npz')
patalogy_data = load(file_dir,'echemia_features_filtered_reduced.npz')

# Створимо датафрейм, закодуємо норму як 1, паталогію, як 0
[joining_mas,flags_mas] = dl.class_coding(patalogy_data,normal_data,clases_description =['паталогія','норма'])

#Нормуємо данні
scaler = StandardScaler()
coefs = scaler.fit(joining_mas)

#збереження
file_dir = 'Features_calculated/standart_scaler_coefs'
file_name = 'standart_scaler_coefs.pkl'
name = pjoin(file_dir,file_name)
pickle.dump(coefs, open(name,'wb'))
# Завантаження
coefs_loaded = pickle.load(open(name,'rb'))
# Нормування
scaled_data =coefs_loaded.transform(joining_mas)

#------------------------Рекурсивне виділення----------------
#формуємо датафрейм
dtt = np.concatenate((joining_mas,flags_mas),axis = 1)
coln = ['(i) normalA', '(i) sttosp', '(i) sttossig', '(i) S_st_segm', '(ii) normalA',
         '(ii) sttosp', '(ii) sttossig', '(ii) S_st_segm', '(iii) normalA',
         '(iii) sttosp', '(iii) sttossig', '(iii) S_st_segm', '(avr) normalA',
         '(avr) sttosp', '(avr) sttossig', '(avr) S_st_segm', '(avl) normalA',
         '(avl) sttosp', '(avl) sttossig', '(avl) S_st_segm', '(avf) normalA',
         '(avf) sttosp', '(avf) sttossig', '(avf) S_st_segm','class']
df = pd.DataFrame(data=dtt,columns=coln)
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.datasets import make_friedman1
#вирізаємо 100 записів з датафрейм
df = df.loc[10300:10500]
x = df.drop(['class'],axis=1)
y = df['class']

if False:
    #x, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
    estimator = SVR(kernel="linear")
    selector = RFE(estimator, n_features_to_select=12, step=10)
    selector = selector.fit(x, y)
    #Рекурсивне виділення
    best_features = selector.support_
    features_line = coln[:-1]
    features_line = np.array(features_line)
    print('Best features:')
    print(features_line[best_features])
    print(selector.support_)
    exit(0)
if False:
    #-----------------------корелційний аналіз---------------
    dataframe = pd.DataFrame(data=scaled_data,columns=colums_names)
    print(dataframe)
    sns.heatmap(dataframe.corr(),annot = True,fmt='.0g');
    plt.show()
    exit(0)

if True:
    #----------------------PCA аналіз------------------
    from sklearn.decomposition import PCA
    pca_breast = PCA(n_components=3)
    print(np.shape(scaled_data))
    principalComponents_breast = pca_breast.fit_transform(scaled_data)
    print('Components information ratio:')
    print(pca_breast.explained_variance_ratio_)
    print('total_information:')
    print(np.sum(pca_breast.explained_variance_ratio_))
    print(np.shape(principalComponents_breast))
    #ploting
    import plotly.express as px
    df = pd.DataFrame({'First_component':principalComponents_breast[:,0],
                       'Second_component':principalComponents_breast[:,1],
                       'Third_component':principalComponents_breast[:,2],'flags':flags_mas[:,0]})
    fig = px.scatter_3d(df, x='First_component', y='Second_component', z='Third_component',color='flags')
    fig.show()

#--------------------------TSNE аналіз---------------------------------------------

if False:
    from sklearn.manifold import TSNE

    #Може рахувати до 20 хвилин!

    # # 3d
    # X_embedded = TSNE(n_components=3, verbose=1).fit_transform(scaled_data)
    # import plotly.express as px
    # df = pd.DataFrame({'First_component': X_embedded[:, 0],
    #                    'Second_component': X_embedded[:, 1],
    #                    'Third_component': X_embedded[:, 2], 'flags': flags_mas[:, 0]})
    # fig = px.scatter_3d(df, x='First_component', y='Second_component', z='Third_component', color='flags')
    # fig.show()

    #2d
    X_embedded = TSNE(n_components=2, verbose=1).fit_transform(scaled_data)
    import plotly.express as px

    df = pd.DataFrame({'First_component': X_embedded[:, 0],
                       'Second_component': X_embedded[:, 1], 'flags': flags_mas[:, 0]})
    fig = px.scatter(df, x='First_component', y='Second_component',  color='flags')
    fig.show()
