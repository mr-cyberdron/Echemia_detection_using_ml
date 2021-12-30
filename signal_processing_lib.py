import numpy as np # підключення NUMPY
from scipy.fft import fft,rfft, fftfreq
from scipy import signal
from os.path import dirname, join as pjoin
import scipy.io as sio
import matplotlib.pyplot as plt


#----------------------------SOS фільтр---------------------------
#files_path повинен мати посилання на папку з файлами
#sosmatr.mat - матриця коефіцієнів sos фільтру, збережені за допомогою do_filter.m нвзва матриці в мат файлі - 'a'
#scaleval.mat - матриця коефіцієнів масштабу , збережені за допомогою do_filter.m нвзва матриці в мат файлі - 'b'

def sos_filter_matlab(sig,files_path):
    sosmatr_path = files_path+'sosmatr.mat'
    scaleval_mat_path = files_path+'scaleval.mat'
    sosmatrix = sio.loadmat(sosmatr_path)['a']
    scalevalues = sio.loadmat(scaleval_mat_path)['b']
    sosmatrix = np.ascontiguousarray(sosmatrix, dtype=np.float32)
    scalevalues = np.ascontiguousarray(scalevalues, dtype=np.float32)
    x_sos = signal.sosfilt(sosmatrix,sig) * np.prod(scalevalues)  # x is my signal
    return (x_sos)

#----------------------------Перетворення фур'є------------------
# Вхід:
# sig
# Fs
# fromfreq
# tofreq
#Вихід:
# fft_sig = # Амплітудний спектр
# Presponce =# Спектр потужності
# f_vector # вектор частот
# normalA = # сумарна відносна амплітуда ділянки
# normalP =  # сумарна відносна потужність ділянки

def fft_transf(sig,Fs,fromfreq,tofreq,plotflag):
    #sig = sig1
    #Fs = Fs
    #fromfreq = 0
    #tofreq= 200#Fs/2

    # вектор часу
    T = len(sig)/Fs
    f_vector = np.linspace(0,Fs/2,int(np.around((Fs/2)*T))) # вектор частот


    # перерахуємо в відліки вектора чатот
    from_count = int(np.around(fromfreq * T))
    to_count = int(np.around(tofreq * T))

    # обраховуємо спектр
    fft_sig = np.abs(rfft(sig-np.mean(sig)))# Амплітудний спектр
    tmp = rfft(sig- np.mean(sig))
    Presponce = np.abs((np.power(tmp,2))/np.size(sig))# Спектр потужності
    maxPresponse = np.sum(Presponce) #сумарна потужність спектру для визначенн індексів
    maxAresponse = np.sum(fft_sig)
    # Виділяємо діапазон в діапазоні частот
    fft_sig = fft_sig[from_count:to_count]
    nA = np.sum(fft_sig) # сумарна потужність спектру ділянки сигналу
    Presponce = Presponce[from_count:to_count]
    nP = np.sum(Presponce)# сумарна амплітуда  спектру ділянки сигналу
    f_vector = f_vector[from_count:to_count]

    # Отримуємо нормовані спектри
    normalAresponce = fft_sig/maxAresponse
    normalPresponce = Presponce/maxPresponse

    normalA = np.sum(normalAresponce)# сумарна відносна амплітуда ділянки
    normalP = np.sum(normalPresponce) # сумарна відносна потужність ділянки

    # повертаємо вектор значень

    if plotflag == 1:
        plt.bar(f_vector, fft_sig)
        plt.title('FFt')  # заголовок
        plt.xlabel("Частота, гц")  # ось абсцисс
        plt.ylabel("Амплітуда, мВ")  # ось ординат
        plt.grid()  # включение отображение сетки
        plt.show()

    return [fft_sig,Presponce,f_vector,normalA,normalP]