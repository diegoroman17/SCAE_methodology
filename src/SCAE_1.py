"""
Stacked Convolutional AutoEncoders
 __author__ = "Prof. Diego Cabrera"
__email__ = "dcabrera@ups.edu.ec"
"""
#import matplotlib
#matplotlib.use("Agg") 
#import matplotlib.pyplot as plt
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
    """Stacked Convolutional Auto-encoder class (SCAE)

    A stacked convolutional autoencoder model is obtained by stacking several
    CAEs. The hidden layer of the CAE at layer `i` becomes the input of
    the CAE at layer `i+1`. The first layer CAE gets as input the input of
    the SCAE, and the hidden layer of the last CAE is connected to dAE.
    Note that after pretraining, the SCAE is dealt with as a normal CNN,
    the CAEs are only used to initialize the weights.
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

        w, h = n_ins
        for i in xrange(self.n_layers):

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

        self.finetune_cost_p = self.logLayer.negative_log_likelihood(self.y)

        self.errors_p = self.logLayer.errors(self.y)
        self.berrors_p = self.logLayer.berrors(self.y)
        
        self.linLayer = LinearRegression(
            input=self.logLayer.p_y_given_x,
            n_in=n_outs,
            n_out=1
        )

        self.params.extend(self.linLayer.params)

        self.finetune_cost = self.linLayer.negative_log_likelihood(self.yr)

        self.errors = self.linLayer.errors(self.yr)
        
        self.predictions = self.linLayer.predictions(self.yr)

    def pretraining_functions(self, train_set_x, batch_size):

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


def test_SCAE(finetune_lr=0.1, pretraining_epochs=100,
             pretrain_lr=0.01, training_epochs=400,
             dataset='mnist.pkl.gz', batch_size=200,
             path_results='/home/titan/Dropbox/Seville/data/results5/',
             extend_verbose=True):

    print 'load data'
    datasets = load_data2(dataset)
    
    train_set_x, train_set_y, train_set_yr = datasets[0]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size

    # numpy random generator
    # start-snippet-3
    numpy_rng = numpy.random.RandomState(89677)
    print '... building the model'
    # construct the stacked denoising autoencoder class
    kernel_layers_sizes=[100, 200]
    kernel_sizes=[(5, 5), (3, 3)]
    hidden_layer_num = 1000
    corruption_levels = [.1, .2, .3]
    
    scae = SCAE(
        numpy_rng=numpy_rng,
        n_ins=(136,64),
        kernel_layers_sizes=kernel_layers_sizes,
        kernel_sizes=kernel_sizes,
        batch_size=batch_size,
        n_outs=10,
        hidden_layer_num = hidden_layer_num,
        extend_verbose=extend_verbose
    )
    # end-snippet-3 start-snippet-4
    with open(path_results + 'parameters', 'w') as f:
        f.write('Architecture:\n')
        f.write('nfilters = ' + str(kernel_layers_sizes) + '\n')
        f.write('filters shape = ' + str(kernel_sizes) + '\n')
        f.write('fully hidden neurons number = ' + str(hidden_layer_num) + '\n\n')
        
        f.write('Pre-training:\n')
        f.write('pretraining learning rate = ' + str(pretrain_lr) + '\n')
        f.write('pretraining epochs = ' + str(pretraining_epochs) + '\n')
        f.write('corruption levels = ' + str(corruption_levels) + '\n\n')
        
        f.write('Training:\n')
        f.write('finetuning learning rate = ' + str(finetune_lr) + '\n')
        f.write('training epochs = ' + str(training_epochs) + '\n')
        f.write('batch size = ' + str(batch_size) + '\n')
    #scae = cPickle.load(open('pretrained_model.pkl'))
    #########################
    # PRETRAINING THE MODEL #
    #########################
    
    print '... getting the pretraining functions'
    
    if extend_verbose:
        pretraining_fns, pretraining_eval = scae.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size)
    else:
        pretraining_fns = scae.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size)
    
    print '... pre-training the model'
    start_time = timeit.default_timer()
    ## Pre-train layer-wise   
    try: 
        os.makedirs(path_results)
    except OSError:
        if not os.path.isdir(path_results):
            raise
    cost_each_layer = []
    for i in xrange(scae.n_layers):
        # go through pretraining epochs
        cost_each_epoch = []
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                val_c = pretraining_fns[i](index=batch_index,
                         corruption=corruption_levels[i],
                         lr=pretrain_lr)
                c.append(val_c)
                
            mean_value = numpy.mean(c)
            cost_each_epoch.append(mean_value)                           
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print mean_value
        cost_each_layer.append(cost_each_epoch)
    
    if extend_verbose:
        for i in xrange(scae.n_layers):
            results_eval = pretraining_eval[i](index=1)
            save2file(path_results + 'pretrain_eval_l'+str(i)+'_b1.pkl.gz', results_eval)
                           
    cost_each_epoch = []
    for epoch in xrange(pretraining_epochs):
        # go through the training set
        c = []
        for batch_index in xrange(n_train_batches):
            val_c = pretraining_fns[-1](index=batch_index,
                    corruption=corruption_levels[-1],
                    lr=pretrain_lr)
            c.append(val_c)
            
        mean_value = numpy.mean(c)
        cost_each_epoch.append(mean_value)
        print 'Pre-training layer final, epoch %d, cost ' % epoch,
        print mean_value
    cost_each_layer.append(cost_each_epoch)
    
    save2file(path_results + 'pretrain_cost.pkl.gz', cost_each_layer)
    end_time = timeit.default_timer()

    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    
    # save the best model
    with open(path_results+'pretrained_model.pkl', 'w') as f:
        cPickle.dump(scae, f)
    # end-snippet-4
    
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
    """ Path to DataSet and path for save results """
    test_SCAE(dataset='/home/titan/Dropbox/Seville/data/hgbdcr_TF_ALL.pkl.gz',
              path_results='/home/titan/Dropbox/Seville/data/results5/')
