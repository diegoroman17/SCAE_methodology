from SCAE_1 import SCAE
import cPickle
import theano
import theano.tensor as T
from Linear_Layer import LinearLayer
from linear_sgd import load_data2
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error, mean_absolute_error
import pylab as pl

dataset='/home/titan/Dropbox/Seville/data/hgbdcr_TF_ALL.pkl.gz'
print 'load data'
batch_size = 200
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

index = T.lscalar()  # index to a [mini]batch

# generate symbolic variables for input (x and y represent a
# minibatch)
x = T.matrix('x')  # data, presented as rasterized images
y = T.vector('y')  # labels, presented as 1D vector of [int] labels


path = '/home/titan/Dropbox/Seville/data/results3/'
model = 'predict_model.pkl'
scae = cPickle.load(open(path+model))

W = np.array([1.0,0.8841762,0.8042414,0.7194127,0.6280587,
              0.4828711,0.3947798,0.2985318,0.1435563,0.0]).transpose()
Lin = LinearLayer(input=scae.logLayer.p_y_given_x,n_in=10,n_out=1,W=W)


test_model = theano.function(
            [index],
            [Lin.y_pred,y, Lin.ydisc],
            givens={
                scae.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                y: test_set_yr[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='test'
        )
yd = []
ypv = []
yiv = []
ydv = []
for i in xrange(n_test_batches):
    yp, yi, ydisc = test_model(i)
    ypv.extend(yp)
    yiv.extend(yi)
    yd.extend(ydisc)

for i in yd:
    ydv.append(W[i])

yiv = np.array(yiv)
ypv = np.array(ypv)
ydv = np.array(ydv)
ydd = np.zeros(ypv.shape)
for i,y in enumerate(ypv):
    idx2 = (np.abs(W-y)).argmin()
    ydd[i] = W[idx2]

idx = np.argsort(yiv)
yiv = yiv[idx]
ypv = ypv[idx]
ydv = ydv[idx]
ydd = ydd[idx]

print mean_squared_error(yiv, ypv)
print mean_squared_error(yiv, ydv)
print mean_squared_error(yiv, ydd)

print r2_score(yiv, ypv)
print r2_score(yiv, ydv)
print r2_score(yiv, ydd)

print median_absolute_error(yiv, ypv)
print median_absolute_error(yiv, ydv)
print median_absolute_error(yiv, ydd)

print mean_absolute_error(yiv, ypv)
print mean_absolute_error(yiv, ydv)
print mean_absolute_error(yiv, ydd)
"""
npts = 1500

pl.plot(yiv[0:npts],'r')
pl.hold('on')
pl.plot(ydv[0:npts],'g')
#pl.plot(ydv(0:npts),'b')
pl.savefig(path+'reg3.pdf')
"""