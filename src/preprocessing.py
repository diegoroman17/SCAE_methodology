import scipy.io as sio
import os
import numpy
import gzip
import pickle as cPickle
from scipy import signal
import matplotlib.pyplot as plt
from pylab import *
import pywt
from sklearn.decomposition import PCA
import scipy.io.wavfile as wavfile
from numpy import fft

# Find the highest power of two less than or equal to the input.
def lepow2(x):
    return 2 ** floor(log2(x))

# Make a scalogram given an MRA tree.
def scalogram(data):
    bottom = 0

    vmin = min(map(lambda x: min(abs(x)), data))
    vmax = max(map(lambda x: max(abs(x)), data))

    gca().set_autoscale_on(True)

    for row in range(0, len(data)):
        scale = 2.0 ** (row - len(data))

        imshow(
            array([abs(data[row])]),
            interpolation = 'nearest',
            vmin = vmin,
            vmax = vmax,
            extent = [0, 150, bottom, bottom + scale])

        bottom += scale

def build_dataset_t_series(path_in=None,
                           file_out=None, 
                           n_cuts=10,
                           sensors = 0, 
                           patterns=None, 
                           percents=(0.7,0.15,0.15)):
    
    train_p, valid_p, test_p = percents
    n_train_p = int(train_p * n_cuts)
    n_valid_p = int(valid_p * n_cuts)
    n_test_p = n_cuts - n_valid_p - n_train_p
    
    train_set_x = [];
    train_set_y = [];
    val_set_x = [];
    val_set_y = [];
    test_set_x = [];
    test_set_y = [];
    
    for pattern in patterns:
        name, new_name = pattern
        path_patt = os.path.join(path_in,name)
        for mfile in os.listdir(path_patt):
            print(mfile)
            m_data = sio.loadmat(os.path.join(path_patt,mfile))
            m_data = m_data['data']
            #fs = m_data['SampleF'][0,0][0][0]
            measures = m_data['Measures'][0,0][sensors]
            wind = int(measures.shape[1] / n_cuts)
            for i in range(0,n_train_p):
                frame = measures[:,i*wind:(i+1)*wind]
                train_set_x.append(numpy.ndarray.tolist(frame.flatten()))
                train_set_y.append(new_name)
                
            for i in range(n_train_p,n_train_p + n_valid_p):
                frame = measures[:,i*wind:(i+1)*wind]
                val_set_x.append(numpy.ndarray.tolist(frame.flatten()))
                val_set_y.append(new_name)
                
            for i in range(n_train_p + n_valid_p,n_cuts):
                frame = measures[:,i*wind:(i+1)*wind]
                test_set_x.append(numpy.ndarray.tolist(frame.flatten()))
                test_set_y.append(new_name)
                
    train_set_x = numpy.array(train_set_x,'float32')
    val_set_x = numpy.array(val_set_x,'float32')  
    test_set_x = numpy.array(test_set_x,'float32')
    train_set_y = numpy.array(train_set_y)
    val_set_y = numpy.array(val_set_y)
    test_set_y = numpy.array(test_set_y)
    
    indices = numpy.random.permutation(train_set_y.shape[0])
    train_set_x = train_set_x[indices,:]
    train_set_y = train_set_y[indices]
    
    indices = numpy.random.permutation(val_set_y.shape[0])
    val_set_x = val_set_x[indices,:]
    val_set_y = val_set_y[indices]
    
    indices = numpy.random.permutation(test_set_y.shape[0])
    test_set_x = test_set_x[indices,:]
    test_set_y = test_set_y[indices]
    
    train_set = train_set_x, train_set_y
    val_set = val_set_x, val_set_y
    test_set = test_set_x, test_set_y

    dataset = [train_set, val_set, test_set]

    f = gzip.open(file_out,'wb')
    cPickle.dump(dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
            

def build_dataset_time_freq(path_in=None,
                           file_out=None, 
                           n_cuts=10,
                           sensors = 0, 
                           patterns=None, 
                           percents=(0.7,0.15,0.15)):
    
    train_p, valid_p, test_p = percents
    n_train_p = int(train_p * n_cuts)
    n_valid_p = int(valid_p * n_cuts)
    n_test_p = n_cuts - n_valid_p - n_train_p
    
    train_set_x = [];
    train_set_y = [];
    val_set_x = [];
    val_set_y = [];
    test_set_x = [];
    test_set_y = [];
    
    level_d = 6
    
    for pattern in patterns:
        name, new_name = pattern
        path_patt = os.path.join(path_in,name)
        for mfile in os.listdir(path_patt):
            print(mfile)
            m_data = sio.loadmat(os.path.join(path_patt,mfile))
            m_data = m_data['data']
            #fs = m_data['SampleF'][0,0][0][0]
            measures = m_data['Measures'][0,0][sensors]
            wind = int(measures.shape[1] / n_cuts)
            wind = lepow2(wind)
            for i in range(0,n_train_p):
                frame = measures[:,i*wind:(i+1)*wind]
                wptree = pywt.WaveletPacket(numpy.ndarray.tolist(frame.flatten()), 'db5','sym',level_d)
                level = wptree.get_level(level_d, order = "freq")
                wavmatr = []
                for node in level:
                    wavmatr.extend(node.data)
                #wavmatr = numpy.array(wavmatr)
                #plt.imshow(wavmatr, extent=[0, 136, 0, 64], cmap='PRGn', aspect='auto',vmax=abs(wavmatr).max(), vmin=0)
                #plt.show()
                train_set_x.append(wavmatr)
                train_set_y.append(new_name)
                
            for i in range(n_train_p,n_train_p + n_valid_p):
                frame = measures[:,i*wind:(i+1)*wind]
                wptree = pywt.WaveletPacket(numpy.ndarray.tolist(frame.flatten()), 'db5','sym',level_d)
                level = wptree.get_level(level_d, order = "freq")
                wavmatr = []
                for node in level:
                    wavmatr.extend(node.data)
                val_set_x.append(wavmatr)
                val_set_y.append(new_name)
                
            for i in range(n_train_p + n_valid_p,n_cuts):
                frame = measures[:,i*wind:(i+1)*wind]
                wptree = pywt.WaveletPacket(numpy.ndarray.tolist(frame.flatten()), 'db5','sym',level_d)
                level = wptree.get_level(level_d, order = "freq")
                wavmatr = []
                for node in level:
                    wavmatr.extend(node.data)
                test_set_x.append(wavmatr)
                test_set_y.append(new_name)
                
    train_set_x = numpy.array(train_set_x)
    val_set_x = numpy.array(val_set_x)  
    test_set_x = numpy.array(test_set_x)
    train_set_y = numpy.array(train_set_y)
    val_set_y = numpy.array(val_set_y)
    test_set_y = numpy.array(test_set_y)
    
    indices = numpy.random.permutation(train_set_y.shape[0])
    train_set_x = train_set_x[indices,:]
    train_set_y = train_set_y[indices]
    
    indices = numpy.random.permutation(val_set_y.shape[0])
    val_set_x = val_set_x[indices,:]
    val_set_y = val_set_y[indices]
    
    indices = numpy.random.permutation(test_set_y.shape[0])
    test_set_x = test_set_x[indices,:]
    test_set_y = test_set_y[indices]
    
    train_set = train_set_x, train_set_y
    val_set = val_set_x, val_set_y
    test_set = test_set_x, test_set_y

    dataset = [train_set, val_set, test_set]

    f = gzip.open(file_out,'wb')
    cPickle.dump(dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    
def build_dataset_time_freq2(path_in=None,
                           file_out=None, 
                           n_cuts=10,
                           sensors = 0, 
                           patterns=None, 
                           percents=(0.7,0.15,0.15)):
    
    train_p, valid_p, test_p = percents
    n_train_p = int(train_p * n_cuts)
    n_valid_p = int(valid_p * n_cuts)
    n_test_p = n_cuts - n_valid_p - n_train_p
    
    train_set_x = [];
    train_set_y = [];
    train_set_yr = [];
    val_set_x = [];
    val_set_y = [];
    val_set_yr = [];
    test_set_x = [];
    test_set_y = [];
    test_set_yr = [];
    
    level_d = 6
    
    for pattern in patterns:
        name, new_name, new_name_r = pattern
        path_patt = os.path.join(path_in,name)
        for mfile in os.listdir(path_patt):
            print(mfile)
            m_data = sio.loadmat(os.path.join(path_patt,mfile))
            m_data = m_data['data']
            #fs = m_data['SampleF'][0,0][0][0]
            measures = m_data['Measures'][0,0][sensors]
            wind = int(measures.shape[1] / n_cuts)
            wind = lepow2(wind)
            for i in range(0,n_train_p):
                frame = measures[:,i*wind:(i+1)*wind]
                wptree = pywt.WaveletPacket(numpy.ndarray.tolist(frame.flatten()), 'db5','sym',level_d)
                level = wptree.get_level(level_d, order = "freq")
                wavmatr = []
                for node in level:
                    wavmatr.extend(node.data)
                #wavmatr = numpy.array(wavmatr)
                #plt.imshow(wavmatr, extent=[0, 136, 0, 64], cmap='PRGn', aspect='auto',vmax=abs(wavmatr).max(), vmin=0)
                #plt.show()
                train_set_x.append(wavmatr)
                train_set_y.append(new_name)
                train_set_yr.append(new_name_r)
                
            for i in range(n_train_p,n_train_p + n_valid_p):
                frame = measures[:,i*wind:(i+1)*wind]
                wptree = pywt.WaveletPacket(numpy.ndarray.tolist(frame.flatten()), 'db5','sym',level_d)
                level = wptree.get_level(level_d, order = "freq")
                wavmatr = []
                for node in level:
                    wavmatr.extend(node.data)
                val_set_x.append(wavmatr)
                val_set_y.append(new_name)
                val_set_yr.append(new_name_r)
                
            for i in range(n_train_p + n_valid_p,n_cuts):
                frame = measures[:,i*wind:(i+1)*wind]
                wptree = pywt.WaveletPacket(numpy.ndarray.tolist(frame.flatten()), 'db5','sym',level_d)
                level = wptree.get_level(level_d, order = "freq")
                wavmatr = []
                for node in level:
                    wavmatr.extend(node.data)
                test_set_x.append(wavmatr)
                test_set_y.append(new_name)
                test_set_yr.append(new_name_r)
                
    train_set_x = numpy.array(train_set_x)
    val_set_x = numpy.array(val_set_x)  
    test_set_x = numpy.array(test_set_x)
    train_set_y = numpy.array(train_set_y)
    val_set_y = numpy.array(val_set_y)
    test_set_y = numpy.array(test_set_y)
    train_set_yr = numpy.array(train_set_yr)
    val_set_yr = numpy.array(val_set_yr)
    test_set_yr = numpy.array(test_set_yr)
    
    indices = numpy.random.permutation(train_set_y.shape[0])
    train_set_x = train_set_x[indices,:]
    train_set_y = train_set_y[indices]
    train_set_yr = train_set_yr[indices]
    
    indices = numpy.random.permutation(val_set_y.shape[0])
    val_set_x = val_set_x[indices,:]
    val_set_y = val_set_y[indices]
    val_set_yr = val_set_yr[indices]
    
    indices = numpy.random.permutation(test_set_y.shape[0])
    test_set_x = test_set_x[indices,:]
    test_set_y = test_set_y[indices]
    test_set_yr = test_set_yr[indices]
    
    train_set = train_set_x, train_set_y, train_set_yr
    val_set = val_set_x, val_set_y, val_set_yr
    test_set = test_set_x, test_set_y, test_set_yr
    

    dataset = [train_set, val_set, test_set]

    f = gzip.open(file_out,'wb')
    cPickle.dump(dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

def build_dataset_time_pca(path_in=None,
                                 file_out=None,
                                 n_cuts=10,
                                 sensors=0,
                                 patterns=None,
                                 percents=(0.7, 0.15, 0.15)):


    train_p, valid_p, test_p = percents
    n_train_p = int(train_p * n_cuts)
    n_valid_p = int(valid_p * n_cuts)
    n_test_p = n_cuts - n_valid_p - n_train_p

    train_set_x = [];
    train_set_y = [];
    train_set_yr = [];
    val_set_x = [];
    val_set_y = [];
    val_set_yr = [];
    test_set_x = [];
    test_set_y = [];
    test_set_yr = [];

    level_d = 6

    for pattern in patterns:
        name, new_name, new_name_r = pattern
        path_patt = os.path.join(path_in, name)
        for mfile in os.listdir(path_patt):
            print(mfile)
            m_data = sio.loadmat(os.path.join(path_patt, mfile))
            m_data = m_data['data']
            # fs = m_data['SampleF'][0,0][0][0]
            measures = m_data['Measures'][0, 0][sensors]
            wind = int(measures.shape[1] / n_cuts)
            wind = lepow2(wind)
            for i in range(0, n_train_p):
                frame = measures[:, i * wind:(i + 1) * wind]
                train_set_x.append(numpy.ndarray.tolist(frame.flatten()))
                train_set_y.append(new_name)
                train_set_yr.append(new_name_r)

            for i in range(n_train_p, n_train_p + n_valid_p):
                frame = measures[:, i * wind:(i + 1) * wind]
                val_set_x.append(numpy.ndarray.tolist(frame.flatten()))
                val_set_y.append(new_name)
                val_set_yr.append(new_name_r)

            for i in range(n_train_p + n_valid_p, n_cuts):
                frame = measures[:, i * wind:(i + 1) * wind]
                test_set_x.append(numpy.ndarray.tolist(frame.flatten()))
                test_set_y.append(new_name)
                test_set_yr.append(new_name_r)

    train_set_x = numpy.array(train_set_x)
    val_set_x = numpy.array(val_set_x)
    test_set_x = numpy.array(test_set_x)
    pca = PCA(n_components=0.95)
    train_set_x = pca.fit_transform(train_set_x)
    val_set_x = pca.transform(val_set_x)
    test_set_x = pca.transform(test_set_x)

    train_set_y = numpy.array(train_set_y)
    val_set_y = numpy.array(val_set_y)
    test_set_y = numpy.array(test_set_y)
    train_set_yr = numpy.array(train_set_yr)
    val_set_yr = numpy.array(val_set_yr)
    test_set_yr = numpy.array(test_set_yr)

    indices = numpy.random.permutation(train_set_y.shape[0])
    train_set_x = train_set_x[indices, :]
    train_set_y = train_set_y[indices]
    train_set_yr = train_set_yr[indices]

    indices = numpy.random.permutation(val_set_y.shape[0])
    val_set_x = val_set_x[indices, :]
    val_set_y = val_set_y[indices]
    val_set_yr = val_set_yr[indices]

    indices = numpy.random.permutation(test_set_y.shape[0])
    test_set_x = test_set_x[indices, :]
    test_set_y = test_set_y[indices]
    test_set_yr = test_set_yr[indices]

    train_set = train_set_x, train_set_y, train_set_yr
    val_set = val_set_x, val_set_y, val_set_yr
    test_set = test_set_x, test_set_y, test_set_yr

    dataset = [train_set, val_set, test_set]

    f = gzip.open(file_out, 'wb')
    cPickle.dump(dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

def build_dataset_freq_pca(path_in=None,
                                 file_out=None,
                                 n_cuts=10,
                                 sensors=0,
                                 patterns=None,
                                 percents=(0.7, 0.15, 0.15)):


    train_p, valid_p, test_p = percents
    n_train_p = int(train_p * n_cuts)
    n_valid_p = int(valid_p * n_cuts)
    n_test_p = n_cuts - n_valid_p - n_train_p

    train_set_x = [];
    train_set_y = [];
    train_set_yr = [];
    val_set_x = [];
    val_set_y = [];
    val_set_yr = [];
    test_set_x = [];
    test_set_y = [];
    test_set_yr = [];

    level_d = 6

    for pattern in patterns:
        name, new_name, new_name_r = pattern
        path_patt = os.path.join(path_in, name)
        for mfile in os.listdir(path_patt):
            print(mfile)
            m_data = sio.loadmat(os.path.join(path_patt, mfile))
            m_data = m_data['data']
            # fs = m_data['SampleF'][0,0][0][0]
            measures = m_data['Measures'][0, 0][sensors]
            wind = int(measures.shape[1] / n_cuts)
            wind = lepow2(wind)
            for i in range(0, n_train_p):
                frame = measures[:, i * wind:(i + 1) * wind]
                frame = numpy.absolute(fft.rfft(frame)/len(frame))**2
                train_set_x.append(numpy.ndarray.tolist(frame.flatten()))
                train_set_y.append(new_name)
                train_set_yr.append(new_name_r)

            for i in range(n_train_p, n_train_p + n_valid_p):
                frame = measures[:, i * wind:(i + 1) * wind]
                frame = numpy.absolute(fft.rfft(frame) / len(frame)) ** 2
                val_set_x.append(numpy.ndarray.tolist(frame.flatten()))
                val_set_y.append(new_name)
                val_set_yr.append(new_name_r)

            for i in range(n_train_p + n_valid_p, n_cuts):
                frame = measures[:, i * wind:(i + 1) * wind]
                frame = numpy.absolute(fft.rfft(frame) / len(frame)) ** 2
                test_set_x.append(numpy.ndarray.tolist(frame.flatten()))
                test_set_y.append(new_name)
                test_set_yr.append(new_name_r)

    train_set_x = numpy.array(train_set_x)
    val_set_x = numpy.array(val_set_x)
    test_set_x = numpy.array(test_set_x)
    pca = PCA(n_components=780)
    train_set_x = pca.fit_transform(train_set_x)
    val_set_x = pca.transform(val_set_x)
    test_set_x = pca.transform(test_set_x)

    train_set_y = numpy.array(train_set_y)
    val_set_y = numpy.array(val_set_y)
    test_set_y = numpy.array(test_set_y)
    train_set_yr = numpy.array(train_set_yr)
    val_set_yr = numpy.array(val_set_yr)
    test_set_yr = numpy.array(test_set_yr)

    indices = numpy.random.permutation(train_set_y.shape[0])
    train_set_x = train_set_x[indices, :]
    train_set_y = train_set_y[indices]
    train_set_yr = train_set_yr[indices]

    indices = numpy.random.permutation(val_set_y.shape[0])
    val_set_x = val_set_x[indices, :]
    val_set_y = val_set_y[indices]
    val_set_yr = val_set_yr[indices]

    indices = numpy.random.permutation(test_set_y.shape[0])
    test_set_x = test_set_x[indices, :]
    test_set_y = test_set_y[indices]
    test_set_yr = test_set_yr[indices]

    train_set = train_set_x, train_set_y, train_set_yr
    val_set = val_set_x, val_set_y, val_set_yr
    test_set = test_set_x, test_set_y, test_set_yr

    dataset = [train_set, val_set, test_set]

    f = gzip.open(file_out, 'wb')
    cPickle.dump(dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

def build_dataset_freq(path_in=None,
                                 file_out=None,
                                 n_cuts=10,
                                 sensors=0,
                                 patterns=None,
                                 percents=(0.7, 0.15, 0.15)):


    train_p, valid_p, test_p = percents
    n_train_p = int(train_p * n_cuts)
    n_valid_p = int(valid_p * n_cuts)
    n_test_p = n_cuts - n_valid_p - n_train_p

    train_set_x = [];
    train_set_y = [];
    train_set_yr = [];
    val_set_x = [];
    val_set_y = [];
    val_set_yr = [];
    test_set_x = [];
    test_set_y = [];
    test_set_yr = [];

    level_d = 6

    for pattern in patterns:
        name, new_name, new_name_r = pattern
        path_patt = os.path.join(path_in, name)
        for mfile in os.listdir(path_patt):
            print(mfile)
            m_data = sio.loadmat(os.path.join(path_patt, mfile))
            m_data = m_data['data']
            # fs = m_data['SampleF'][0,0][0][0]
            measures = m_data['Measures'][0, 0][sensors]
            wind = int(measures.shape[1] / n_cuts)
            wind = lepow2(wind)
            for i in range(0, n_train_p):
                frame = measures[:, i * wind:(i + 1) * wind]
                frame = numpy.absolute(fft.rfft(frame)/len(frame))**2
                train_set_x.append(numpy.ndarray.tolist(frame.flatten()))
                train_set_y.append(new_name)
                train_set_yr.append(new_name_r)

            for i in range(n_train_p, n_train_p + n_valid_p):
                frame = measures[:, i * wind:(i + 1) * wind]
                frame = numpy.absolute(fft.rfft(frame) / len(frame)) ** 2
                val_set_x.append(numpy.ndarray.tolist(frame.flatten()))
                val_set_y.append(new_name)
                val_set_yr.append(new_name_r)

            for i in range(n_train_p + n_valid_p, n_cuts):
                frame = measures[:, i * wind:(i + 1) * wind]
                frame = numpy.absolute(fft.rfft(frame) / len(frame)) ** 2
                test_set_x.append(numpy.ndarray.tolist(frame.flatten()))
                test_set_y.append(new_name)
                test_set_yr.append(new_name_r)

    train_set_x = numpy.array(train_set_x)
    val_set_x = numpy.array(val_set_x)
    test_set_x = numpy.array(test_set_x)


    train_set_y = numpy.array(train_set_y)
    val_set_y = numpy.array(val_set_y)
    test_set_y = numpy.array(test_set_y)
    train_set_yr = numpy.array(train_set_yr)
    val_set_yr = numpy.array(val_set_yr)
    test_set_yr = numpy.array(test_set_yr)

    indices = numpy.random.permutation(train_set_y.shape[0])
    train_set_x = train_set_x[indices, :]
    train_set_y = train_set_y[indices]
    train_set_yr = train_set_yr[indices]

    indices = numpy.random.permutation(val_set_y.shape[0])
    val_set_x = val_set_x[indices, :]
    val_set_y = val_set_y[indices]
    val_set_yr = val_set_yr[indices]

    indices = numpy.random.permutation(test_set_y.shape[0])
    test_set_x = test_set_x[indices, :]
    test_set_y = test_set_y[indices]
    test_set_yr = test_set_yr[indices]

    train_set = train_set_x, train_set_y, train_set_yr
    val_set = val_set_x, val_set_y, val_set_yr
    test_set = test_set_x, test_set_y, test_set_yr

    dataset = [train_set, val_set, test_set]

    f = gzip.open(file_out, 'wb')
    cPickle.dump(dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

def build_dataset_tf_PCA(path_in=None,
                           file_out=None,
                           n_cuts=10,
                           sensors = 0,
                           patterns=None,
                           percents=(0.7,0.15,0.15)):

    train_p, valid_p, test_p = percents
    n_train_p = int(train_p * n_cuts)
    n_valid_p = int(valid_p * n_cuts)
    n_test_p = n_cuts - n_valid_p - n_train_p

    train_set_x = [];
    train_set_y = [];
    train_set_yr = [];
    val_set_x = [];
    val_set_y = [];
    val_set_yr = [];
    test_set_x = [];
    test_set_y = [];
    test_set_yr = [];

    level_d = 6

    for pattern in patterns:
        name, new_name, new_name_r = pattern
        path_patt = os.path.join(path_in,name)
        for mfile in os.listdir(path_patt):
            print(mfile)
            m_data = sio.loadmat(os.path.join(path_patt,mfile))
            m_data = m_data['data']
            #fs = m_data['SampleF'][0,0][0][0]
            measures = m_data['Measures'][0,0][sensors]
            wind = int(measures.shape[1] / n_cuts)
            wind = lepow2(wind)
            for i in range(0,n_train_p):
                frame = measures[:,i*wind:(i+1)*wind]
                wptree = pywt.WaveletPacket(numpy.ndarray.tolist(frame.flatten()), 'db5','sym',level_d)
                level = wptree.get_level(level_d, order = "freq")
                wavmatr = []
                for node in level:
                    wavmatr.extend(node.data)
                #wavmatr = numpy.array(wavmatr)
                #plt.imshow(wavmatr, extent=[0, 136, 0, 64], cmap='PRGn', aspect='auto',vmax=abs(wavmatr).max(), vmin=0)
                #plt.show()
                train_set_x.append(wavmatr)
                train_set_y.append(new_name)
                train_set_yr.append(new_name_r)

            for i in range(n_train_p,n_train_p + n_valid_p):
                frame = measures[:,i*wind:(i+1)*wind]
                wptree = pywt.WaveletPacket(numpy.ndarray.tolist(frame.flatten()), 'db5','sym',level_d)
                level = wptree.get_level(level_d, order = "freq")
                wavmatr = []
                for node in level:
                    wavmatr.extend(node.data)
                val_set_x.append(wavmatr)
                val_set_y.append(new_name)
                val_set_yr.append(new_name_r)

            for i in range(n_train_p + n_valid_p,n_cuts):
                frame = measures[:,i*wind:(i+1)*wind]
                wptree = pywt.WaveletPacket(numpy.ndarray.tolist(frame.flatten()), 'db5','sym',level_d)
                level = wptree.get_level(level_d, order = "freq")
                wavmatr = []
                for node in level:
                    wavmatr.extend(node.data)
                test_set_x.append(wavmatr)
                test_set_y.append(new_name)
                test_set_yr.append(new_name_r)

    train_set_x = numpy.array(train_set_x)
    val_set_x = numpy.array(val_set_x)
    test_set_x = numpy.array(test_set_x)

    pca = PCA(n_components=0.95)
    train_set_x = pca.fit_transform(train_set_x)
    val_set_x = pca.transform(val_set_x)
    test_set_x = pca.transform(test_set_x)

    train_set_y = numpy.array(train_set_y)
    val_set_y = numpy.array(val_set_y)
    test_set_y = numpy.array(test_set_y)
    train_set_yr = numpy.array(train_set_yr)
    val_set_yr = numpy.array(val_set_yr)
    test_set_yr = numpy.array(test_set_yr)

    indices = numpy.random.permutation(train_set_y.shape[0])
    train_set_x = train_set_x[indices,:]
    train_set_y = train_set_y[indices]
    train_set_yr = train_set_yr[indices]

    indices = numpy.random.permutation(val_set_y.shape[0])
    val_set_x = val_set_x[indices,:]
    val_set_y = val_set_y[indices]
    val_set_yr = val_set_yr[indices]

    indices = numpy.random.permutation(test_set_y.shape[0])
    test_set_x = test_set_x[indices,:]
    test_set_y = test_set_y[indices]
    test_set_yr = test_set_yr[indices]

    train_set = train_set_x, train_set_y, train_set_yr
    val_set = val_set_x, val_set_y, val_set_yr
    test_set = test_set_x, test_set_y, test_set_yr


    dataset = [train_set, val_set, test_set]

    f = gzip.open(file_out,'wb')
    cPickle.dump(dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()



if __name__ == '__main__':
    path_in = '/home/diego/Desktop/helical_nomalized'
    file_out = '/home/diego/data/hgbdcr_TF_ALL.pkl'
    n_cuts = 40
    '''
    patterns = [('P1',1.0),('P2',0.8841762),
                ('P3',0.8042414),('P4',0.7194127)]
                '''
    '''
    patterns = [('P1',1.0),('P2',0.8841762),
                ('P3',0.8042414),('P4',0.7194127),
                ('P5',0.6280587),('P6',0.4828711),
                ('P7',0.3947798),('P8',0.2985318),
                ('P9',0.1435563),('P10',0.0)]
    '''

    patterns = [('P1',0,1.0),('P2',1,0.8841762),
                ('P3',2,0.8042414),('P4',3,0.7194127),
                ('P5',4,0.6280587),('P6',5,0.4828711),
                ('P7',6,0.3947798),('P8',7,0.2985318),
                ('P9',8,0.1435563),('P10',9,0.0)]

    #patterns = [('P1', 0, 1.0)]
    train_p = 60.0/100.0;
    valid_p = 20.0/100.0;
    test_p = 1.0 - train_p - valid_p
    percents = train_p, valid_p, test_p
    sensors = [0]
    #build_dataset_t_series(path_in, file_out, n_cuts, sensors, patterns, percents)
    build_dataset_time_freq2(path_in, file_out, n_cuts, sensors, patterns, percents)
    #build_dataset_time_pca(path_in, file_out, n_cuts, sensors, patterns, percents)
    #build_dataset_freq_pca(path_in, file_out, n_cuts, sensors, patterns, percents)
    #build_dataset_freq(path_in, file_out, n_cuts, sensors, patterns, percents)
    #build_dataset_tf_PCA(path_in, file_out, n_cuts, sensors, patterns, percents)
    