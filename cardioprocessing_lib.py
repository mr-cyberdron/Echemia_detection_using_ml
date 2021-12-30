import numpy as np # підключення NUMPY
from scipy import signal
#matplotlib
import matplotlib.pyplot as plt

#-----------------------------Виділення кардіоциклів, та ритмограми-----------------
#На вхід - сигнал, частота дискритезації
#на виході - масиви значень

# налаштування
cof = 10#степінь дифереренціонування, більше число - менша чутливість
firstwindow = 3# Довжина вікна визначення меж порогової обробки в секундах
treshold_value = 0.6# значення порогу спрацювання порогового детектора
delaytime = 60/240# час затримки після спрацювання порогового детектора с
# фільтрація сигналу filterflag = true/false
filter_flag = True
#Параметри  фільтр батерворта
order = 5
cutoff_freq = 30
# параметри виділення кардіоциклів
cof_bef = 0.5#0.40#0.35#0.267  # відношення інтервала PR до сердцевого ритму
cof_aft = 0.95#0.75#0.7#0.55  # відношення інтервала RT до сердцевого ритму

def cardiocicles(sig,Fs):
    #Створення фільтра
    sos = signal.butter(order, cutoff_freq, 'lowpass', fs=Fs, output='sos')
    #фільтрація
    filtered_sig = signal.sosfilt(sos,sig)
    tmpp =[]# тимчасовий масив
    if filter_flag == True:
        # створюємо диференційований сигнал по філььтрованому сигналу
        for i in range(0, filtered_sig.size - 1 - cof, 1):
            tmp = filtered_sig[i + cof] - filtered_sig[i]
            tmpp.append(tmp / cof)  # dy/dt
    else:
        for i in range(0,sig.size-1-cof,1):
            tmp =sig[i+cof] - sig[i]
            tmpp.append(tmp/cof)# dy/dt

    sig_diff = np.array(tmpp) # створюємо масив диференціювання
    calibration_part = sig_diff[1*Fs:(firstwindow+1)*Fs]# відрізок для встановлення порогових значень
    max_value = np.amax(calibration_part) #Максимальне значення
    mean_value = np.mean(calibration_part)#Мінімальне
    treshold = max_value - ((max_value-mean_value)*(1-treshold_value))#розраховуємо поріг спрацювання

    bool_matr = np.zeros((int(sig_diff.size)))# матриця порогової обробки
    i = 0 #Лічильник відліків
    cardio_delay = 0#Лічільник затримки RR інтервалу
    rytmogram = []
    while i < sig_diff.size and i< (sig_diff.size - delaytime*Fs): # проходимо по всім елементам масиву диференціонування
        i = i+1
        cardio_delay = cardio_delay+1 # лічильник RR інтервалу
        if sig_diff[i]>treshold: # якщо спрацьовує поріг, пишемо 1, чекаємо
            rytmogram.append(cardio_delay/Fs) #пишемо значення RR інтервалу
            cardio_delay = 0 #якщо детектується пороговий рівень лічильник обнуляється
            bool_matr[i] = 1
            i = round(i+(delaytime*Fs))
        else:
            pass

    # Виділення кардіоциклів
    #Знаходимо середнью тривалість кардіоциклу
    mean_RR = np.mean(rytmogram)
    #Знайдемо кількість відліків до та після детектування кардіоциклу
    counts_bef = round(mean_RR*cof_bef*Fs)
    counts_aft = round(mean_RR*cof_aft*Fs)
    #створюємо матрицю кардіоциклів
    cardiocicles = []
    for num_elem in range(0,bool_matr.size,1):
        if bool_matr[num_elem] == 1 and num_elem<(sig.size-counts_aft) and num_elem> counts_bef:
           cardiocicle_matr = sig[(num_elem-counts_bef):(num_elem+counts_aft)]
           cardiocicles.append(cardiocicle_matr)

    return (cardiocicles)

#виділення ритмограми
def rytmogramm(sig,Fs):
    #Створення фільтра
    sos = signal.butter(order, cutoff_freq, 'lowpass', fs=Fs, output='sos')
    #фільтрація
    filtered_sig = signal.sosfilt(sos,sig)
    tmpp =[]# тимчасовий масив
    if filter_flag == True:
        # створюємо диференційований сигнал по філььтрованому сигналу
        for i in range(0, filtered_sig.size - 1 - cof, 1):
            tmp = filtered_sig[i + cof] - filtered_sig[i]
            tmpp.append(tmp / cof)  # dy/dt
    else:
        for i in range(0,sig.size-1-cof,1):
            tmp =sig[i+cof] - sig[i]
            tmpp.append(tmp/cof)# dy/dt

    sig_diff = np.array(tmpp) # створюємо масив диференціювання
    calibration_part = sig_diff[1*Fs:(firstwindow+1)*Fs]# відрізок для встановлення порогових значень
    max_value = np.amax(calibration_part) #Максимальне значення
    mean_value = np.mean(calibration_part)#Мінімальне
    treshold = max_value - ((max_value-mean_value)*(1-treshold_value))#розраховуємо поріг спрацювання

    bool_matr = np.zeros((int(sig_diff.size)))# матриця порогової обробки
    i = 0 #Лічильник відліків
    cardio_delay = 0#Лічільник затримки RR інтервалу
    rytmogram = []
    while i < sig_diff.size and i< (sig_diff.size - delaytime*Fs): # проходимо по всім елементам масиву диференціонування
        i = i+1
        cardio_delay = cardio_delay+1 # лічильник RR інтервалу
        if sig_diff[i]>treshold: # якщо спрацьовує поріг, пишемо 1, чекаємо
            rytmogram.append(cardio_delay/Fs) #пишемо значення RR інтервалу
            cardio_delay = 0 #якщо детектується пороговий рівень лічильник обнуляється
            bool_matr[i] = 1
            i = round(i+(delaytime*Fs))
        else:
            pass
    return (rytmogram)

#-----------------------нормалізація сигналу---------------------
def normilize_sig(sig):
    # видалення постійної складової
    dif = 0 - (np.mean(sig))
    norm_sig = []
    for i in range((np.shape(sig)[0])):
        norm_sig.append(sig[i]+dif)
    tmp = norm_sig
    norm_sig = []
    norm_sig = np.divide(tmp,np.max(tmp))
    return (norm_sig)

#-------------------побудова сигналу----------------------
def plott(sig,Fs,title):
    sig = np.array(sig)
    Tvector = np.array(list(range(np.size(sig))))
    Tvector = Tvector * (1 / Fs)
    plt.plot(Tvector,sig)
    plt.title(title)  # заголовок
    plt.xlabel("Час, с")  # ось абсцисс
    plt.ylabel("Амплітуда, мВ")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.show()

#----------------------------виділення сегментів-------------------------
def find_segm(sig):
    from_num_1 = int(np.size(sig) * 0.5)
    to_num_1 = int(np.size(sig))
    st_segm = sig[from_num_1:to_num_1]

    from_num_2 = 0
    to_num_2 = int(np.size(sig) * 0.25)
    pq_segm = sig[from_num_2:to_num_2]

    qs_segm = sig[to_num_2:from_num_1]
    return([pq_segm,qs_segm,st_segm])

