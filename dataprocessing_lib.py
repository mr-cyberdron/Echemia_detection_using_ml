from os.path import dirname, join as pjoin #os - функції для роботи з операційною системою,
import numpy as np # підключення NUMPY
#matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mode as mode
from sklearn.preprocessing import StandardScaler
import pandas as pd
#------------------збереження данних---------------------
def storing_data(data,storage_dir,storage_filename):
    storage_filepath = pjoin(storage_dir, storage_filename)
    # зберіганні зі стисненням
    print('storing ' + storage_filename)
    np.savez_compressed(storage_filepath, data)
    print('done!')
#------------------Завантаження данних---------------------
def load_data(file_dir,file_name):
    print('loaddd ' + file_name)
    fpath = pjoin(file_dir, file_name)  # створюємо шлях до файлу
    dump_array = np.load(fpath, allow_pickle=True)
    array = dump_array['arr_0']
    print('loading done!')
    return (array)
#------------------Вивід гістограми розподілу ознак---------------
def dataplot(*datamas,names = [],numbins = [],title = []):
    if not names:
        autoflag = 1
    else:
        autoflag = 0
    if not numbins:
        bins_autoflag =1
    else:
        bins_autoflag =0
    counter = 0

    for data in datamas:
        if autoflag == 1:
            names.append(str(counter+1))
        if autoflag == 0:
            pass
        if bins_autoflag == 1:
            numbins = ((round((np.shape(data)[0]) / 10)) + 1)
        plt.subplot(1,3,1)
        if not(not title):
            plt.suptitle(title)
        plt.hist(data,alpha=0.5,bins= numbins,label= names[counter])
        plt.legend(loc='upper right')
        plt.title("Гістограма розподілу")
        x1 = list(range(int(np.shape(data)[0])))
        x2 = data[1:len(data)]
        x2 = list(x2)
        x2.append(data[len(data)-1])

        plt.subplot(1, 3, 2)
        plt.scatter(x1,data,alpha=0.5,label = names[counter])
        plt.legend(loc='upper right')
        plt.title("Гістограма розсіювання")
        plt.xlabel("Номер ознаки")
        plt.ylabel("Значення")
        plt.subplot(1, 3, 3)
        plt.scatter(x2, data, alpha=0.5, label=names[counter])
        plt.legend(loc='upper right')
        plt.title("Гістограма розсіювання")
        plt.xlabel("значення n+1")
        plt.ylabel("значення n")
        counter = counter + 1
    plt.show()
#----------видалення викидів---------------------
def emission_remove(data,colums_names,range = [],plotflag = 0, remove = 0,pandasform = 0):
    tmp = pd.DataFrame(data,columns=colums_names)
    data = tmp
    print('starting emission removing')
    if not range:
        range = [0.01,0.99]
    #для кожної ознаки
    #початкова змінна
    df_filtered = data
    counter = 1
    for col in colums_names:
        print(str(counter)+ '/'+ str(len(colums_names)))
        counter = counter+1
        tmp = data[col].to_numpy()
        data_np = tmp.copy()
        # визначаємо краї діапазонів без виидів
        q_low = df_filtered[col].quantile(range[0])
        q_hi = df_filtered[col].quantile(range[1])
        # заміна викидів
        if remove == 0:
            sig_mode = mode(df_filtered[col].to_numpy())[0]
            #замінюємо викиди медіаною
            df_filtered.loc[df_filtered[col]>q_hi,col] =sig_mode
            df_filtered.loc[df_filtered[col] < q_low, col] = sig_mode
            df_filtered.loc[df_filtered[col] == (np.nan),col] = sig_mode
        else:
            #Видалення викидів
            df_filtered = df_filtered[((df_filtered[col] < q_hi)) & (df_filtered[col] > q_low)]
        df_filtered_np = df_filtered[col].to_numpy()
        if plotflag != 0:
            dataplot(data_np,df_filtered_np, names=['Вихідні значення','Відфільтровані значення'], numbins=100,title=col)
    print('removing done')
    if pandasform == 0:
        stored_mas = df_filtered.to_numpy()
    else:
        stored_mas = df_filtered
    return (stored_mas)



def increase_features_num(data, notice_num,plotflag = 0):
    print('Srarting increasing feadures_num')
    tmp_matr = []
    tmp_counter = 0
    for i in range(int(np.shape(data)[1])):
        print(str(i) + '/' + str(np.shape(data)[1]))
        mu = np.mean(data[:,i])
        sigma = np.std(data[:,i])
        num = notice_num-(np.shape(data)[0])
        s = (np.random.normal(mu, sigma, int(num))).reshape(num,1)
        if tmp_counter == 0:
            tmp_matr=s.copy()
            tmp_counter=1
        else:
            tmp_matr = np.concatenate((tmp_matr,s),axis = 1)
    increasing_matr = np.concatenate((data,tmp_matr),axis = 0)
    for i in range(int(np.shape(increasing_matr)[1])):
        if plotflag != 0:
            dataplot(data[:, i], increasing_matr[:,i], names=['Вихідні значення', 'Збільшена кількість значень'],
                        numbins=100, title='Результат збільшення ознак')
        else:
            pass
    print('done')
    return (increasing_matr)

def reduse_features_num(data,notice_num,plotflag = 0):
    print('starting decreasing fetures num')
    if notice_num >= int(np.shape(data)[0]):
        raise Exception('notice num should be < shape of data')

    num_of_decreasing = int(np.shape(data)[0])-notice_num
    decreasing_matr = data.copy()
    for i in range(int(num_of_decreasing)):
        randomInts = np.random.randint(0,np.shape(decreasing_matr)[0])
        decreasing_matr = np.delete(decreasing_matr,[randomInts],axis = 0)
        print( str(i) + '/' + str(num_of_decreasing))

    for i in range(int(np.shape(decreasing_matr)[1])):
        if plotflag != 0:
            dataplot(data[:, i], decreasing_matr[:,i], names=['Вихідні значення', 'Зменшена кількість значень'],
                        numbins=100, title='Результат зменшення ознак')
        else:
            pass
    print('done')
    return (decreasing_matr)

def class_coding(*classesmasives,clases_description = [],colums_names =[],normalization = False,coding = True):
    print('starting class coding')
    first_mas_flag = 0
    flags_counter = 0
    for i in classesmasives:
        if first_mas_flag == 0:
            joining_mas = i.copy()
            first_mas_flag = 1
            flags_mas = np.ones((np.shape(i)[0],1))*flags_counter
            if not clases_description:
                print(str(flags_counter)+':'+str(flags_counter)+'('+ str(np.shape(i)[0])+')')
            else:
                print(str(flags_counter)+ ':'+ str(clases_description[flags_counter])+'('+ str(np.shape(i)[0])+')')
                flags_counter = flags_counter + 1

        else:
            joining_mas = np.concatenate((joining_mas,i),axis = 0)
            tmp = np.ones((np.shape(i)[0], 1))*flags_counter
            flags_mas = np.concatenate((flags_mas,tmp),axis = 0)

            if not clases_description:
                print(str(flags_counter) + ':' + str(flags_counter) + '(' + str(np.shape(i)[0]) + ')')
            else:
                print(
                    str(flags_counter) + ':' + str(clases_description[flags_counter]) + '(' + str(np.shape(i)[0]) + ')')
                flags_counter = flags_counter + 1
    print('Всього:' + str(np.shape(joining_mas)))


    if normalization == True:
        # Нормуємо данні
        scaler = StandardScaler()
        coefs = scaler.fit(joining_mas)
        joining_mas = coefs.transform(joining_mas)
        print('normalize')

    colums_names2 = []
    colums_names2.append('class')
    for i in colums_names:
        colums_names2.append(i)

    colums_names = colums_names2
    # формуємо датафрейм
    dtt = np.concatenate((flags_mas,joining_mas), axis=1)

    df = pd.DataFrame(data=dtt, columns=colums_names)
    return [df, colums_names]


