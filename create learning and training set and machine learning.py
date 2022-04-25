import numpy as np # підключення NUMPY
from dataprocessing_lib import storing_data as store
from dataprocessing_lib import load_data as load
import pandas as pd
import dataprocessing_lib as dl
from sklearn.preprocessing import StandardScaler
import pickle
from os.path import dirname, join as pjoin #os - функції для роботи з операційною системою,
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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
# Створимо датафрейм, закодуємо норму як 1, паталогію, як 0,проведемо нормалізацію
[df,colums_names2] = dl.class_coding(patalogy_data,normal_data,clases_description =['норма','паталогія'],
                                          colums_names = colums_names,normalization = True)
# виділимо необхідні ознаки згідно вибору найінформативніших
#
# X = df[['(i) normalA', '(i) sttosp', '(i) sttossig', '(i) S_st_segm', '(ii) normalA',
#          '(ii) sttosp', '(ii) sttossig', '(ii) S_st_segm', '(iii) normalA',
#          '(iii) sttosp', '(iii) sttossig', '(iii) S_st_segm', '(avr) normalA',
#          '(avr) sttosp', '(avr) sttossig', '(avr) S_st_segm', '(avl) normalA',
#          '(avl) sttosp', '(avl) sttossig', '(avl) S_st_segm', '(avf) normalA',
#          '(avf) sttosp', '(avf) sttossig', '(avf) S_st_segm']]


X = df[['(i) normalA', '(i) sttosp', '(i) sttossig', '(i) S_st_segm']]
y = df[['class']]
#розділимо на тренувальну та тестову вибірку
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
if True:
    if True:
        # #логістична регресія
        # print('starting LogisticRegress')
        # from sklearn.linear_model import LogisticRegression
        # clf = LogisticRegression(random_state=0).fit(X_train, y_train)
        # predicted = clf.predict(X_test[:2, :])
        # print('control',y_test[:2])
        # print('predicted',predicted)
        # scor = clf.score(X_test,y_test)
        # print('score = ', scor)
        # print('done')
        # print('')
        #
        # #лінійна регресія
        # print('starting linearRegress')
        # from sklearn.linear_model import LinearRegression
        # clf = LinearRegression().fit(X_train, y_train)
        # predicted = clf.predict(X_test[:2, :])
        # print('control',y_test[:2])
        # print('predicted',predicted)
        # scor = clf.score(X_test,y_test)
        # print('score = ', scor)
        # print('done')
        # print('')
        #
        # #гребнева регресія
        # print('ridge regression')
        # from sklearn import linear_model
        # clf = linear_model.Ridge(alpha=0.1)
        # clf.fit(X_train, y_train)
        # predicted = clf.predict(X_test[:2, :])
        # print('control',y_test[:2])
        # print('predicted',predicted)
        # scor = clf.score(X_test,y_test)
        # print('score = ', scor)
        # print('done')
        # print('')
        #
        # #ласо регресія
        # print('lasso regression')
        # from sklearn import linear_model
        # clf = linear_model.Lasso(alpha=0.1)
        # clf.fit(X_train, y_train)
        # predicted = clf.predict(X_test[:2, :])
        # print('control',y_test[:2])
        # print('predicted',predicted)
        # scor = clf.score(X_test,y_test)
        # print('score = ', scor)
        # print('done')
        # print('')
        #
        #
        # # Дискримінантний аналіз
        # print('starting LinearDiscriminantAnalysis')
        # from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        # clf = LinearDiscriminantAnalysis()
        # clf.fit(X_train, y_train)
        # predicted = clf.predict(X_test[:2, :])
        # print('control',y_test[:2])
        # print('predicted',predicted)
        # scor = clf.score(X_test,y_test)
        # print('score = ', scor)
        # print('done')
        # print('')
        #
        # # метод k - найближчих сусідів
        # print('starting KNeighborsClassifier n = 3')
        # from sklearn.neighbors import KNeighborsClassifier
        # clf = KNeighborsClassifier(n_neighbors=3)
        # clf.fit(X_train, y_train)
        # predicted = clf.predict(X_test[:2, :])
        # print('control',y_test[:2])
        # print('predicted',predicted)
        # scor = clf.score(X_test,y_test)
        # print('score = ', scor)
        # print('done')
        # print('')
        #
        #
        # ядерний метод опорних векторів
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        #print('ядерний метод опорних векторів, поліноміальне ядро')
        # #метод вимагає стандартного розподілу, тож використовуємо standartscaler
        # clf = make_pipeline(StandardScaler(),SVC(kernel='poly', degree=3,verbose=1))
        # clf.fit(X_train, y_train)
        # predicted = clf.predict(X_test[:2, :])
        # print('control',y_test[:2])
        # print('predicted',predicted)
        # scor = clf.score(X_test,y_test)
        # print('score = ', scor)
        # print('done')
        # print('')
        #

        print('ядерний метод опорних векторів, гаусівське ядро')
        #метод вимагає стандартного розподілу, тож використовуємо standartscaler
        clf = make_pipeline(StandardScaler(), SVC(kernel='rbf',gamma=1,  C = 1000,verbose=1))
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_test[:2, :])
        print('control',y_test[:2])
        print('predicted',predicted)
        scor = clf.score(X_test,y_test)
        print('score = ', scor)
        y_pred = clf.predict(X_test)
        # Get the confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        # Now the normalize the diagonal entries
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        accuracy = cm.diagonal()
        # array([1.        , 0.        , 0.66666667])
        print('accuracy')
        print({'Normal class accuracy':accuracy[0], 'Pathology class accuracy':accuracy[1]})
        print('done')
        print('')


        # print('ядерний метод опорних векторів, сигмоїдальне ядро')
        # #метод вимагає стандартного розподілу, тож використовуємо standartscaler
        # clf = make_pipeline(StandardScaler(), SVC(kernel='sigmoid',verbose=1))
        # clf.fit(X_train, y_train)
        # predicted = clf.predict(X_test[:2, :])
        # print('control',y_test[:2])
        # print('predicted',predicted)
        # scor = clf.score(X_test,y_test)
        # print('score = ', scor)
        # print('done')
        # print('')
        #
        # #Дерева рішень
        # print('Дерева рішень')
        # from sklearn.tree import DecisionTreeClassifier
        # deepmass = [4,7,10]
        # for i in deepmass:
        #     print('deepth:', i)
        #     clf = DecisionTreeClassifier(max_depth=i)
        #     clf.fit(X_train, y_train)
        #     predicted = clf.predict(X_test[:2, :])
        #     print('control',y_test[:2])
        #     print('predicted',predicted)
        #     scor = clf.score(X_test,y_test)
        #     print('score = ', scor)
        #     print('done')
        #     print('')

    #ансамблі дерев рішень
    from sklearn.ensemble import RandomForestClassifier
    print('Ансамблі дерев рішень')
    num_treas = [10,20,30]
    for i in num_treas:
        print('кількість дерев в лісі',i)
        clf = RandomForestClassifier(n_estimators=i)
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_test[:2, :])
        print('control',y_test[:2])
        print('predicted',predicted)
        scor = clf.score(X_test,y_test)
        print('score = ', scor)
        print('done')
        print('')
        y_pred = clf.predict(X_test)
        # Get the confusion matrix
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_test, y_pred)
        # Now the normalize the diagonal entries
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        accuracy = cm.diagonal()
        # array([1.        , 0.        , 0.66666667])
        print('accuracy')
        print({'Normal class accuracy': accuracy[0], 'Pathology class accuracy': accuracy[1]})
        y_pred = clf.predict(X_test)
        # Get the confusion matrix
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_test, y_pred)
        # Now the normalize the diagonal entries
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        accuracy = cm.diagonal()
        # array([1.        , 0.        , 0.66666667])
        print('accuracy')
        print({'Normal class accuracy': accuracy[0], 'Pathology class accuracy': accuracy[1]})
        print('done')
        print('')

    #градієнтний бустинг
    from sklearn.ensemble import  GradientBoostingClassifier
    print('градієнтний бустинг')
    clf = GradientBoostingClassifier(max_depth=10,
                                     learning_rate=1,verbose=1)
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test[:2, :])
    print('control',y_test[:2])
    print('predicted',predicted)
    scor = clf.score(X_test,y_test)
    print('score = ', scor)
    y_pred = clf.predict(X_test)
    # Get the confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    # Now the normalize the diagonal entries
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    accuracy = cm.diagonal()
    # array([1.        , 0.        , 0.66666667])
    print('accuracy')
    print({'Normal class accuracy':accuracy[0], 'Pathology class accuracy':accuracy[1]})
    print('done')
    print('')


        # #Наївний баєсівський ксласифікатор
        # from sklearn.naive_bayes import GaussianNB
        # print('naive bayes clasifier')
        # clf = GaussianNB()
        # clf.fit(X_train, y_train)
        # predicted = clf.predict(X_test[:2, :])
        # print('control',y_test[:2])
        # print('predicted',predicted)
        # scor = clf.score(X_test,y_test)
        # print('score = ', scor)
        # print('done')
        # print('')

#Стекінг (Найкращій )
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import  GradientBoostingClassifier

print('stacking')
estimators = [('rf', RandomForestClassifier(n_estimators=20, random_state=42)),
              ('GBC',GradientBoostingClassifier(max_depth=10, learning_rate=1,verbose=1))]
#estimators = [('rf', RandomForestClassifier(n_estimators=10, random_state=42))]
clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(),verbose=1)

clf.fit(X_train, y_train)
predicted = clf.predict(X_test[:2, :])
print('control',y_test[:2])
print('predicted',predicted)
scor = clf.score(X_test,y_test)
print('score = ', scor)
y_pred = clf.predict(X_test)
# Get the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# Now the normalize the diagonal entries
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
accuracy = cm.diagonal()
# array([1.        , 0.        , 0.66666667])
print('accuracy')
print({'Normal class accuracy':accuracy[0], 'Pathology class accuracy':accuracy[1]})
print('done')
print('')
#cross validation score
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X_train,y_train, cv=5,verbose=1)

print(scores)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


#precision recall
from sklearn.metrics import recall_score,precision_score
predictions = clf.predict(X_test)
recall = recall_score(y_test,predictions)
precision = precision_score(y_test,predictions)

print('Precision:',precision,'Recall:',recall)
from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(y_test, predictions)
from CF_matrix_plotter import make_confusion_matrix
labels = ['True Neg','False Pos','False Neg','True Pos']
categories = ['Normal', 'Pathology']
make_confusion_matrix(cf_matrix,
                      group_names=labels,
                      categories=categories,
                      cmap='Blues',
                      sum_stats=True)
plt.show()


