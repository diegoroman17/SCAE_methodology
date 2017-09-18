import cPickle
import numpy as np
from linear_sgd import load_data2

path = '/home/dcabrera/Dropbox/Seville/data/results3/'

print 'load data'
batch_size = 200
dataset='/home/dcabrera/Dropbox/Seville/data/hgbdcr_TF_ALL.pkl.gz'
datasets = load_data2(dataset)
    
train_set_x, train_set_y, train_set_yr = datasets[0]
valid_set_x, valid_set_y, valid_set_yr = datasets[1]
test_set_x, test_set_y, test_set_yr = datasets[2]

    # compute number of minibatches for training, validation and testing
n_train_batches = train_set_x.get_value(borrow=True).shape[0]
n_train_batches /= batch_size
n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

w, h = (136, 64)
win_size_w, win_size_h = (5, 5)
mask_ini = np.zeros(win_size_w*win_size_h,dtype=np.int32)
for i in range(win_size_h):
    mask_ini[i*win_size_w:(i+1)*win_size_w] = range((i*w),(i*w+win_size_w))

print 'generating probability maps'
output_model = cPickle.load(open(path+'output_model.pkl'))
prob_map = np.zeros((h - win_size_h + 1,w - win_size_w + 1,batch_size,10))
for index in xrange(n_test_batches):
    input_val = test_set_x.get_value(borrow=True)[index * batch_size: (index + 1) * batch_size]
    mask = np.copy(mask_ini)
    for j in range(h - win_size_h + 1):
        for i in range(w - win_size_w + 1):
            input_val_mod = input_val
            input_val_mod[:,mask] = 0
            prob_map[j,i] = output_model(input_val_mod)[0]
            mask += 1
        mask+= (win_size_w - 1)
    with open(path+'prob_map'+str(index)+'.pkl', 'w') as f:
        cPickle.dump(prob_map, f)