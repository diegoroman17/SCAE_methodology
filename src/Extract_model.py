from SCAE_1 import SCAE
import cPickle
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
from linear_sgd import load_data2

path = '/home/titan/Dropbox/Seville/data/results3/'
model = 'predict_model.pkl'
scae = cPickle.load(open(path+model))
print 'load data'
batch_size = 200
dataset='/home/titan/Dropbox/Seville/data/hgbdcr_TF_ALL.pkl.gz'
datasets = load_data2(dataset)
    
train_set_x, train_set_y, train_set_yr = datasets[0]
valid_set_x, valid_set_y, valid_set_yr = datasets[1]
test_set_x, test_set_y, test_set_yr = datasets[2]

    # compute number of minibatches for training, validation and testing
n_train_batches = train_set_x.get_value(borrow=True).shape[0]
n_train_batches /= batch_size
n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

# numpy random generator
# start-snippet-3
numpy_rng = np.random.RandomState(89677)

output_model = theano.function(
            [scae.x],
            [scae.logLayer.p_y_given_x],
            name='test'
        )
# save the best model
with open(path+'output_model.pkl', 'w') as f:
    cPickle.dump(output_model, f)