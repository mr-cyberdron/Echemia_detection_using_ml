from os.path import dirname, join as pjoin #os - функції для роботи з операційною системою,
# os.path - робота з файловою системою
import re # модуль роботи з рядками тексту
import numpy as np # підключення NUMPY
#matplotlib
import matplotlib.pyplot as plt
# инженерные расщеты, работа с
from scipy.fft import fft,rfft, fftfreq
from scipy import signal
import cardioprocessing_lib as cl # модуль роботи з кардіосигналами
import signal_processing_lib as sp


def fiture_extractor(sig1,Fs,plotflag):
    # будуємо вихідний сигнал
    if plotflag == 1:
        cl.plott(sig1, Fs, 'вихідний сигнал')

    # нормалізуємо сигнал
    sig1 = cl.normilize_sig(sig1)
    if plotflag == 1:
        cl.plott(sig1, Fs, 'нормалізований сигнал')
    # фільтруємо сигнал
    mat_files_path = ''
    sig1 = sp.sos_filter_matlab(sig1, mat_files_path)
    # будуємо фільтрований сигнал
    if plotflag == 1:
        cl.plott(sig1, Fs, 'фільтрований сигнал')
    # виділимо сегменти
    [pq_segm, qs_segm, st_segm] = cl.find_segm(sig1)
    if plotflag == 1:
        cl.plott(pq_segm, Fs, 'PQ segment')
        cl.plott(qs_segm, Fs, 'QS segment')
        cl.plott(st_segm, Fs, 'ST segm')
    # перетворення фур'є
    fromfreq = 4
    tofreq = 8  # Fs/2
    [fft_sig, Presponce, f_vector, normalA, normalP] = sp.fft_transf(sig1, Fs, fromfreq, tofreq, plotflag)
    #Обрахуємо площі
    S_pq_segm = np.sum(pq_segm)
    S_qs_segm = np.sum(qs_segm)
    S_st_segm = np.sum(st_segm)
    s_sig = np.sum(sig1)
    #обрахуємо ознаки
    sttosp = (S_st_segm/S_pq_segm)
    sttossig = (S_st_segm/s_sig)

    return ([normalA,sttosp,sttossig,S_st_segm])


# функція збереження метаданних
def feature_dumper(metadata, file_dir, teg, storage_dir, storage_filename):
    # для кожного пацієнта зі списку
    fiture_table_echemia_pat = []
    num_pat = 0  # 0 - немає ешемії
    num_fiture_sig = 0
    for file in metadata:
        num_pat = num_pat + 1
        print('loaddd' + teg + file)
        fpath = pjoin(file_dir, file)  # створюємо шлях до файлу
        dump_array = np.load(fpath, allow_pickle=True)
        # для кожного цікавлючого відведення

        for num_of_lead in range(6):
            cardiocicles = dump_array['arr_0'][num_of_lead]['cardiocicles']
            # для кожного кардіоциклу
            lead_fiture_table = []
            for num_of_fig in range(int(np.shape(cardiocicles)[0])):
                sig1 = cardiocicles[num_of_fig]
                print(teg)
                print('сигнал ' + str(num_pat) + '/' + str((np.shape(metadata)[0])))
                print('Відведення ' + str(num_of_lead + 1) + '/' + '6')
                print('кардіоцикл ' + str(num_of_fig + 1) + '/' + str((np.shape(cardiocicles)[0])))
                print(' ')
                # виділення ознак
                pat_fiture_line = fiture_extractor(sig1, Fs, 0)
                # Заповнюємо рядок
                lead_fiture_table.append(pat_fiture_line)
            # для першого проходу, створюємо масив, записуємо перше відведення
            if num_of_lead == 0:
                pat_fiture_table = lead_fiture_table
            # Для подальших
            else:
                # Якщо один більше іншого, заповнюємо недостаючі поля медіаною
                if int(np.shape(pat_fiture_table)[0]) > int(np.shape(lead_fiture_table)[0]):
                    tmp = np.ones((((np.shape(pat_fiture_table)[0]) - (np.shape(lead_fiture_table)[0])),
                                   (np.shape(lead_fiture_table)[1])))
                    qq = (np.median(lead_fiture_table, axis=0)) * tmp
                    lead_fiture_table = np.concatenate([lead_fiture_table, qq], axis=0)

                elif int(np.shape(pat_fiture_table)[0]) < int(np.shape(lead_fiture_table)[0]):
                    tmp = np.ones((((np.shape(lead_fiture_table)[0]) - (np.shape(pat_fiture_table)[0])),
                                   (np.shape(pat_fiture_table)[1])))
                    qq = (np.median(pat_fiture_table, axis=0)) * tmp
                    pat_fiture_table = np.concatenate([pat_fiture_table, qq], axis=0)
                # поєднуємо відведення
                pat_fiture_table = np.concatenate([np.array(pat_fiture_table), np.array(lead_fiture_table)], axis=1)
        # якщо вперше, то створюємо масив
        if num_pat == 1:
            fiture_table_echemia_pat = pat_fiture_table
        # в іншому випадку поєднуємо пацієнтів в один масив
        else:
            fiture_table_echemia_pat = np.concatenate([fiture_table_echemia_pat, pat_fiture_table], axis=0)
        print(fiture_table_echemia_pat)
        print(np.shape(fiture_table_echemia_pat))

    print(teg + ' DONE')

    # Збереження файлу

    storage_filepath = pjoin(storage_dir, storage_filename)

    # зберіганні зі стисненням
    print('storing fitures of' + teg)
    np.savez_compressed(storage_filepath, fiture_table_echemia_pat)
    print('done!')











#----------------------------------Параметри------------------------
Fs = 1000
leads_name_matr =['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
fitures_names = ['normalA','sttosp','sttossig','S_st_segm']
fiture_table_echemia_norm= []
fiture_table_echemia_pat= []
fiture_description = []
for i in range(6):
    for j in range(4):
        fiture_description.append('('+ leads_name_matr[i] + ') '+fitures_names[j])
fiture_description.append('ichemia_flag')
#print(fiture_description)
#-----------------------------Завантажуємо пацієнтів з нормою------------
# 120 кардиоиклов 15 отведений, 80 сигналов
#Fs = 1000
# mv
# ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'vx', 'vy', 'vz'
#
# array['arr_0']
# 	|
#         v
#
# 1 ->    'original'
# 	      'cardiocicles'
# 	       'rytmogram'
#
# 2 ->    'original'
# 	      'cardiocicles'
# 	      'rytmogram'
#
# .
# .
# .
# 16 ->[i,avr....](metadata)
#Завантажуємо список пацієнтів з нормою
pat_flag = 0
print('loaddd metadata of norm')
file = 'storage_data.npz'
file_dir = 'F:\_bases\PhisioNETdatabase cardiopatology\PREPEARED DATA\ecg_normal'
fpath = pjoin(file_dir, file) # створюємо шлях до файлу
dump_array = np.load(fpath,allow_pickle=True)
tmp = dump_array['arr_0']
# список пацієнтів з нормою
norm_metadata = []
for i in range (np.shape(tmp)[0]):
    norm_metadata.append((tmp[i,0])+'.npz')


# Налаштування
metadata = norm_metadata
file_dir = 'F:\_bases\PhisioNETdatabase cardiopatology\PREPEARED DATA\ecg_normal'
teg = 'norm'
storage_dir = 'Features_calculated'
storage_filename = 'NORMAL_features'
#Зберігаємо ознаки
feature_dumper(metadata,file_dir,teg,storage_dir,storage_filename)


#--------------------------------Завантажуємо пацієнтів з паталогією----------
#Завантажуємо список
print('loaddd metadata with pat')
file = 'storage_data.npz'
file_dir = 'F:\_bases\PhisioNETdatabase cardiopatology\PREPEARED DATA\ecg_pat_eshemia_ptb'
fpath = pjoin(file_dir, file) # створюємо шлях до файлу
dump_array = np.load(fpath,allow_pickle=True)
tmp = dump_array['arr_0']
pat_metadata = []
for i in range (np.shape(tmp)[0]):
    pat_metadata.append((tmp[i])+'.npz')


# Налаштування
metadata = pat_metadata
file_dir = 'F:\_bases\PhisioNETdatabase cardiopatology\PREPEARED DATA\ecg_pat_eshemia_ptb'
teg = 'pat'
storage_dir = 'Features_calculated'
storage_filename = 'ESHEMIA_features'
# Зберігаємо ознаки
feature_dumper(metadata,file_dir,teg,storage_dir,storage_filename)











