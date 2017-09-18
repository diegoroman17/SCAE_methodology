import cPickle
#import time
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from mlp import HiddenLayer
from logistic_sgd import LogisticRegression
from linear_sgd import LinearRegression, load_data2
from convolutional_mlp import LeNetConvPoolLayer
from CAE import CAE
from dA import dA
from utils import save2file#, tile_raster_images
#import copy


# start-snippet-1
class SCAE(object):
    """Stacked denoising auto-encoder class (SdA)

    A stacked denoising autoencoder model is obtained by stacking several
    dAs. The hidden layer of the dA at layer `i` becomes the input of
    the dA at layer `i+1`. The first layer dA gets as input the input of
    the SdA, and the hidden layer of the last dA represents the output.
    Note that after pretraining, the SdA is dealt with as a normal MLP,
    the dAs are only used to initialize the weights.
    """

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        n_ins=(28,28),
        batch_size = 20,
        kernel_layers_sizes=[5, 5],
        kernel_sizes=[(5, 5), (5, 5)],
        hidden_layer_num = 100,
        n_outs=10,
        corruption_levels=[0.1, 0.1],
        extend_verbose=True
    ):
        """ This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the sdA

        :type n_layers_sizes: list of ints
        :param n_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network

        :type corruption_levels: list of float
        :param corruption_levels: amount of corruption to use for each
                                  layer
        """
        self.extend_verbose = extend_verbose
        self.conv_layers = []
        self.CAE_layers = []
        self.params = []
        self.n_layers = len(kernel_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y') # the labels are presented as 1D vector of
                                # [int] labels
        # end-snippet-1
        self.yr = T.vector('yr')
        #self.x = x.reshape((batch_size, 1, n_ins[0], n_ins[1]))
        
        # The SdA is an MLP, for which all weights of intermediate layers
        # are shared with a different denoising autoencoders
        # We will first construct the SdA as a deep multilayer perceptron,
        # and when constructing each sigmoidal layer we also construct a
        # denoising autoencoder that shares weights with that layer
        # During pretraining we will train these autoencoders (which will
        # lead to chainging the weights of the MLP as well)
        # During finetunining we will finish training the SdA by doing
        # stochastich gradient descent on the MLP
        # start-snippet-2
        w, h = n_ins
        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            w_shape, h_shape = kernel_sizes[i]
            if i == 0:
                image_shape = (batch_size, 1, w, h)
                filter_shape = (kernel_layers_sizes[0], 1, w_shape, h_shape)
            else:
                w = numpy.int((w - w_shape + 1) / 2)
                h = numpy.int((h - h_shape + 1) / 2)
                image_shape = (batch_size, kernel_layers_sizes[i-1], w, h)
                filter_shape = (kernel_layers_sizes[i], kernel_layers_sizes[i-1], w_shape, h_shape)

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x.reshape((batch_size, 1, n_ins[0], n_ins[1]))
            else:
                layer_input = self.conv_layers[-1].output

            conv_layer = LeNetConvPoolLayer(numpy_rng=numpy_rng,
                                        input=layer_input,
                                        image_shape=image_shape,
                                        filter_shape=filter_shape)
            # add the layer to our list of layers
            self.conv_layers.append(conv_layer)
            # its arguably a philosophical question...
            # but we are going to only declare that the parameters of the
            # sigmoid_layers are parameters of the StackedDAA
            # the visible biases in the dA are parameters of those
            # dA, but not the SdA
            self.params.extend(conv_layer.params)

            # Construct a denoising autoencoder that shared weights with this
            # layer
            CAE_layer = CAE(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          image_shape=image_shape,
                          filter_shape=filter_shape,
                          W=conv_layer.W,
                          bhid=conv_layer.b)
            self.CAE_layers.append(CAE_layer)
        # end-snippet-2
        # We now need to add a logistic layer on top of the MLP
        w_shape, h_shape = kernel_sizes[-1]
        w = numpy.int((w - w_shape + 1) / 2)
        h = numpy.int((h - h_shape + 1) / 2)
        self.sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=self.conv_layers[-1].output.flatten(2),
                                        n_in=kernel_layers_sizes[-1] * w * h,
                                        n_out=hidden_layer_num,
                                        activation=T.nnet.sigmoid)
        self.params.extend(self.sigmoid_layer.params)
        
        self.dA_layer = dA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=self.conv_layers[-1].output.flatten(2),
                          n_visible=kernel_layers_sizes[-1] * w * h,
                          n_hidden=hidden_layer_num,
                          W=self.sigmoid_layer.W,
                          bhid=self.sigmoid_layer.b)
        
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layer.output,
            n_in=hidden_layer_num,
            n_out=n_outs
        )

        self.params.extend(self.logLayer.params)
        self.params_p = list(self.params)#self.params#copy.deepcopy(self.params)
        # construct a function that implements one step of finetunining

        # compute the cost for second phase of training,
        # defined as the negative log likelihood
        self.finetune_cost_p = self.logLayer.negative_log_likelihood(self.y)
        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors_p = self.logLayer.errors(self.y)
        self.berrors_p = self.logLayer.berrors(self.y)
        
        self.linLayer = LinearRegression(
            input=self.logLayer.p_y_given_x,
            n_in=n_outs,
            n_out=1
        )

        self.params.extend(self.linLayer.params)
        # construct a function that implements one step of finetunining

        # compute the cost for second phase of training,
        # defined as the negative log likelihood
        self.finetune_cost = self.linLayer.negative_log_likelihood(self.yr)
        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.linLayer.errors(self.yr)
        
        self.predictions = self.linLayer.predictions(self.yr)

    def pretraining_functions(self, train_set_x, batch_size):
        ''' Generates a list of functions, each of them implementing one
        step in trainnig the dA corresponding to the layer with same index.
        The function will require as input the minibatch index, and to train
        a dA you just need to iterate, calling the corresponding function on
        all minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared variable that contains all datapoints used
                            for training the dA

        :type batch_size: int
        :param batch_size: size of a [mini]batch

        :type learning_rate: float
        :param learning_rate: learning rate used during training for any of
                              the dA layers
        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for CAE in self.CAE_layers:
            # get the cost and the updates list
            cost, updates = CAE.get_cost_updates(corruption_level=corruption_level,
                                                learning_rate=learning_rate)
            # compile the theano function
            fn = theano.function(
                inputs=[
                    index,
                    theano.Param(corruption_level, default=0.2),
                    theano.Param(learning_rate, default=0.1)
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin: batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)
            
        cost, updates = self.dA_layer.get_cost_updates(corruption_level=corruption_level,
                                                learning_rate=learning_rate)
        # compile the theano function
        fn = theano.function(
            inputs=[
                index,
                theano.Param(corruption_level, default=0.2),
                theano.Param(learning_rate, default=0.1)
            ],
            outputs=cost,
            updates=updates,
            givens={
                self.x: train_set_x[batch_begin: batch_end]
            }
        )
            # append `fn` to the list of functions
        pretrain_fns.append(fn)
        
        if self.extend_verbose:
            pretrain_eval = []
            for CAE in self.CAE_layers:
                # get the cost and the updates list
                out = CAE.get_evaluation()
                # compile the theano function
                fn = theano.function(
                    inputs=[index],
                    outputs=out,
                    givens={
                        self.x: train_set_x[batch_begin: batch_end]
                    }
                )
                # append `fn` to the list of functions
                pretrain_eval.append(fn)
            return pretrain_fns, pretrain_eval
                
        return pretrain_fns
    
    def build_finetune_functions_p(self, datasets, batch_size, learning_rate):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on
        a batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                         the has to contain three pairs, `train`,
                         `valid`, `test` in this order, where each pair
                         is formed of two Theano variables, one for the
                         datapoints, the other for the labels

        :type batch_size: int
        :param batch_size: size of a minibatch

        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage
        '''

        (train_set_x, train_set_y, train_set_yr) = datasets[0]
        (valid_set_x, valid_set_y, valid_set_yr) = datasets[1]
        (test_set_x, test_set_y, test_set_yr) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost_p, self.params_p)

        # compute list of fine-tuning updates
        updates = [
            (param, param - gparam * learning_rate)
            for param, gparam in zip(self.params_p, gparams)
        ]

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost_p,
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='train'
        )

        test_score_i = theano.function(
            [index],
            self.errors_p,
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: test_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='test'
        )
        
        test_result_i = theano.function(
            [index],
            self.berrors_p,
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: test_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='testresult'
        )
        
        valid_score_i = theano.function(
            [index],
            self.errors_p,
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: valid_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='valid'
        )
        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]
        
        # Create a function that scans the entire test set
        def test_result():
            return [test_result_i(i) for i in xrange(n_test_batches)]
        
        return train_fn, valid_score, test_score, test_result
    
    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on
        a batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                         the has to contain three pairs, `train`,
                         `valid`, `test` in this order, where each pair
                         is formed of two Theano variables, one for the
                         datapoints, the other for the labels

        :type batch_size: int
        :param batch_size: size of a minibatch

        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage
        '''

        (train_set_x, train_set_y, train_set_yr) = datasets[0]
        (valid_set_x, valid_set_y, valid_set_yr) = datasets[1]
        (test_set_x, test_set_y, test_set_yr) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = [
            (param, param - gparam * learning_rate)
            for param, gparam in zip(self.params, gparams)
        ]

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.yr: train_set_yr[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='train'
        )

        test_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.yr: test_set_yr[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='test'
        )

        valid_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.yr: valid_set_yr[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='valid'
        )
        
        test_predictions_i = theano.function(
            [index],
            self.predictions,
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.yr: test_set_yr[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='test_pred'
        )

        valid_predictions_i = theano.function(
            [index],
            self.predictions,
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.yr: valid_set_yr[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='valid_pred'
        )

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]
        
        # Create a function that scans the entire validation set
        def valid_predictions():
            return [valid_predictions_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_predictions():
            return [test_predictions_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score, valid_predictions, test_predictions


def test_SCAE(finetune_lr=0.1,
             training_epochs=400,
             dataset='mnist.pkl.gz', batch_size=200,
             path_results='/home/titan/Dropbox/Seville/data/resultsCNN1/',
             #path_results='/home/diego/Dropbox/Seville/data/results1/',
              repetitions = 10,
             extend_verbose=True):
    """
    Demonstrates how to train and test a stochastic denoising autoencoder.

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used in the finetune stage
    (factor for the stochastic gradient)

    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining

    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training

    :type n_iter: int
    :param n_iter: maximal number of iterations ot run the optimizer

    :type dataset: string
    :param dataset: path the the pickled dataset

    """

    print 'load data'
    datasets = load_data2(dataset)   
    train_set_x, train_set_y, train_set_yr = datasets[0]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size

    # numpy random generator
    # start-snippet-3
    numpy_rng = numpy.random.RandomState(None)
    print '... building the model'
    # construct the stacked denoising autoencoder class
    kernel_layers_sizes=[100, 200]
    kernel_sizes=[(5, 5), (3, 3)]
    for test_number in range(repetitions):
        path_results += 'test'+str(test_number)+'/'
        if not os.path.exists(path_results):
            os.makedirs(path_results)
        scae = SCAE(
            numpy_rng=numpy_rng,
            n_ins=(136,64),
            kernel_layers_sizes=kernel_layers_sizes,
            kernel_sizes=kernel_sizes,
            batch_size=batch_size,
            hidden_layer_num = 1000,
            n_outs=10,
            extend_verbose=extend_verbose
        )
        # end-snippet-3 start-snippet-4

        with open(path_results + 'parameters', 'w') as f:
            f.write('Architecture:\n')
            f.write('nfilters = ' + str(kernel_layers_sizes) + '\n')
            f.write('filters shape = ' + str(kernel_sizes) + '\n')
            f.write('fully hidden neurons number = ' + str(kernel_layers_sizes) + '\n\n')
            f.write('Training:\n')
            #f.write('pretraining learning rate = ' + str(pretrain_lr) + '\n')
            #f.write('pretraining epochs = ' + str(pretraining_epochs) + '\n')
            f.write('finetuning learning rate = ' + str(finetune_lr) + '\n')
            f.write('training epochs = ' + str(training_epochs) + '\n')
            f.write('batch size = ' + str(batch_size) + '\n')
        ########################
        # FINETUNING THE MODEL #
        ########################

        # get the training, validation and testing function for the model
        print '... getting the finetuning functions'
        train_fn_p, validate_model_p, test_model_p, test_result_p = scae.build_finetune_functions_p(
            datasets=datasets,
            batch_size=batch_size,
            learning_rate=finetune_lr
        )

        print '... finetunning the predictor model'
        # early-stopping parameters
        patience = 10 * n_train_batches  # look as this many examples regardless
        patience_increase = 2.  # wait this much longer when a new best is
                                # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(n_train_batches, patience / 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch

        best_validation_loss = numpy.inf
        test_score = 0.
        start_time = timeit.default_timer()

        done_looping = False
        epoch = 0

        train_cost = []
        val_cost = []
        test_cost = []

        while (epoch < training_epochs) and (not done_looping):
            inc_epoch = epoch * n_train_batches
            for minibatch_index in xrange(n_train_batches):
                #prueba = prueba_model(minibatch_index)
                minibatch_avg_cost = train_fn_p(minibatch_index)
                itera = inc_epoch + minibatch_index
                train_cost.append((itera, epoch, minibatch_avg_cost))
                #print('minibatch %i/%i, cost %f ' %
                #      (minibatch_index + 1, n_train_batches,
                #       minibatch_avg_cost))


                if (itera + 1) % validation_frequency == 0:
                    validation_losses = validate_model_p()
                    this_validation_loss = numpy.mean(validation_losses)
                    val_cost.append((itera, epoch, this_validation_loss))
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch + 1, minibatch_index + 1, n_train_batches,
                           this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        #improve patience if loss improvement is good enough
                        if (
                            this_validation_loss < best_validation_loss *
                            improvement_threshold
                        ):
                            patience = max(patience, itera * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = itera

                        # test it on the test set
                        test_losses = test_model_p()
                        test_score = numpy.mean(test_losses)
                        test_cost.append((itera, epoch, test_score))
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch + 1, minibatch_index + 1, n_train_batches,
                               test_score * 100.))
                        # save the best model
                        with open(path_results + 'predict_model.pkl', 'w') as f:
                            cPickle.dump(scae, f)
                        save2file(path_results + 'test_result.pkl.gz', test_result_p())
                if patience <= itera:
                    done_looping = True
                    break
            epoch = epoch + 1
        save2file(path_results + 'train_cost.pkl.gz', (train_cost, val_cost, test_cost))

        end_time = timeit.default_timer()
        print(
            (
                'Optimization complete with best validation score of %f %%, '
                'on iteration %i, '
                'with test performance %f %%'
            )
            % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
        )
        print >> sys.stderr, ('The training code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))

    

if __name__ == '__main__':
    #test_SCAE(dataset='/home/dcabrera/Dropbox/Seville/data/hgbdcr_TF_ALL.pkl.gz')
    test_SCAE(dataset='/home/titan/Dropbox/Seville/data/hgbdcr_TF_ALL.pkl.gz')
