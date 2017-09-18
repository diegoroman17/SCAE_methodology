"""
Convolutional AutoEncoder Layer
 __author__ = "Prof. Diego Cabrera"
__email__ = "dcabrera@ups.edu.ec"
"""

import os
import sys
import timeit

import numpy
#import matplotlib.pyplot as plt

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import load_data
from utils import tile_raster_images
#from matplotlib.pyplot import imshow

"""
try:
    import PIL.Image as Image
except ImportError:
    import Image
"""

class CAE(object):
    """Convolutional Denoising Auto-Encoder class (CAE)

    A denoising autoencoders tries to reconstruct the input from a corrupted
    version of it by projecting it first in a latent space and reprojecting
    it afterwards back in the input space. Please refer to Vincent et al.,2008
    for more details. If x is the input then equation (1) computes a partially
    destroyed version of x by means of a stochastic mapping q_D. Equation (2)
    computes the projection of the input into the latent space. Equation (3)
    computes the reconstruction of the input, while equation (4) computes the
    reconstruction error. The same principle is applied to convolutional layers.

    """

    def __init__(
        self,
        numpy_rng,
        theano_rng,
        input,
        image_shape,
        filter_shape,
        poolsize=(2, 2),
        W=None,
        bhid=None,
        bvis=None
    ):
        """
        Initialize the dA class by specifying the number of visible units (the
        dimension d of the input ), the number of hidden units ( the dimension
        d' of the latent or hidden space ) and the corruption level. The
        constructor also receives symbolic variables for the input, weights and
        bias. Such a symbolic variables are useful when, for example the input
        is the result of some computations, or when weights are shared between
        the dA and an MLP layer. When dealing with SdAs this always happens,
        the dA on layer 2 gets as input the output of the dA on layer 1,
        and the weights of the dA are used in the second stage of training
        to construct an MLP.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                     generated based on a seed drawn from `rng`

        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for
                      standalone dA

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
                  shared belong the dA and another architecture; if dA should
                  be standalone set this to None

        :type bhid: theano.tensor.TensorType
        :param bhid: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None

        :type bvis: theano.tensor.TensorType
        :param bvis: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None
                     
        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)


        """
        assert image_shape[1] == filter_shape[1]

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            # there are "num input feature maps * filter height * filter width"
            # inputs to each hidden unit
            fan_in = numpy.prod(filter_shape[1:])
            # each unit in the lower layer receives a gradient from:
            # "num output feature maps * filter height * filter width" /
            #   pooling size
            fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
            # initialize weights with random weights
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(
                    (filter_shape[1],),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=numpy.zeros(
                    (filter_shape[0],),
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.dimshuffle(1, 0, 2, 3)[:, :, ::-1, ::-1]
        self.theano_rng = theano_rng
        self.poolsize = poolsize
        # if no input is given, generate a variable representing the input
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input
        
        #self.output = self.get_hidden_values(self.x)
        self.params = [self.W, self.b, self.b_prime]

    def get_corrupted_input(self, input, corruption_level):
        """This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``coruption_level``
        Note : first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
               second argument is the number of trials
               third argument is the probability of success of any trial

                this will produce an array of 0s and 1s where 1 has a
                probability of 1 - ``corruption_level`` and 0 with
                ``corruption_level``

                The binomial function return int64 data type by
                default.  int64 multiplicated by the input
                type(floatX) always return float64.  To keep all data
                in floatX when floatX is float32, we set the dtype of
                the binomial to floatX. As in our case the value of
                the binomial is always 0 or 1, this don't change the
                result. This is needed to allow the gpu to work
                correctly as it only support float32 for now.

        """
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input
        #return input

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            border_mode='valid'
        )
        
        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=self.poolsize,
            ignore_border=True
        )
        
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        return T.tanh(pooled_out+ self.b.dimshuffle('x', 0, 'x', 'x'))

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer

        """
        unpooled_out = T.extra_ops.repeat(T.extra_ops.repeat(hidden, self.poolsize[0],axis=2), self.poolsize[1], axis=3)
        
        deconv_out = conv.conv2d(
            input=unpooled_out,
            filters=self.W_prime,
            border_mode='full'
        )
        return T.tanh(deconv_out+ self.b_prime.dimshuffle('x', 0, 'x', 'x'))

    def get_cost_updates(self, corruption_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        #E = T.mean(T.mean(T.mean(T.power((self.x - z),2),axis=1),axis=1),axis=1)#/(self.x.shape[1]*(self.x.shape[2]*self.x.shape[3]))

        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        #L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        #cost = -T.mean(self.x * T.log(T.abs_(z)) + (1 - self.x) * T.log(T.abs_(1 - z)))
        cost = T.mean(T.pow(self.x - z,2))

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]
        return (cost, updates)
    
    def get_evaluation(self):
        y = self.get_hidden_values(self.x)
        return self.x, y, self.get_reconstructed_input(y)




def test_CAE(learning_rate=0.1, training_epochs=10,
            dataset='mnist.pkl.gz',
            batch_size=20, output_folder='dA_plots'):

    """
    This demo is tested on MNIST

    :type learning_rate: float
    :param learning_rate: learning rate used for training the DeNosing
                          AutoEncoder

    :type training_epochs: int
    :param training_epochs: number of epochs used for training

    :type dataset: string
    :param dataset: path to the picked dataset

    """
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # start-snippet-2
    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    # end-snippet-2

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    ####################################
    # BUILDING THE MODEL NO CORRUPTION #
    ####################################

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    
    layer0_input = x.reshape((batch_size, 1, 28, 28))
    
    
    da = CAE(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 28, 28),
        filter_shape=(5, 1, 5, 5),
        poolsize=(2, 2)
    )

    cost, updates = da.get_cost_updates(
        corruption_level=0.3,
        learning_rate=learning_rate
    )

    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    images = da.get_evaluation()
    evaluate_da = theano.function(
        [index],
        images,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )
    start_time = timeit.default_timer()

    ############
    # TRAINING #
    ############
    fig = plt.figure()
    # go through training epochs
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))

        print('Training epoch %d, cost ' % epoch, numpy.mean(c))
        im_or, im_proc = evaluate_da(0)
        im1 = im_or[0][0]
        im2 = im_proc[0][0]
        
        a=fig.add_subplot(1,2,1) 
        imshow(im1)
    
        a=fig.add_subplot(1,2,2) 
        imshow(im2)
        plt.show()

    end_time = timeit.default_timer()

    training_time = (end_time - start_time)

    print >> sys.stderr, ('The no corruption code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((training_time) / 60.))
    
    


if __name__ == '__main__':
    theano.config.openmp = True
    #theano.config.exception_verbosity = 'high'
    #theano.config.compute_test_value = 'warn'
    test_CAE(dataset='/home/diego/DeepLearningTutorials/data/mnist.pkl.gz')
