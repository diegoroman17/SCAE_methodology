from sklearn import svm
from sklearn.model_selection import StratifiedKFold
import pickle as cPickle
import gzip
import os
import numpy
import timeit

def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    print('... loading data')

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    train_set_x, train_set_y, _ = train_set
    val_set_x, val_set_y, _ = valid_set
    test_set_x, test_set_y, _ = test_set
    data = numpy.vstack((train_set_x, val_set_x, test_set_x))
    target = numpy.hstack((train_set_y, val_set_y, test_set_y))
    #data,target = shared_dataset((data,target))

    rval = data,target
    return rval


dataset='/home/dcabrera/Dropbox/Seville/data/hgbdcr_TF_ALL.pkl'
path_results='/home/dcabrera/Dropbox/Seville/data/kfolds/resultsSVM/'

print('load data')
data,target = load_data(dataset)
print('Finish load data')
numpy_rng = numpy.random.RandomState(None)
skf = StratifiedKFold(n_splits=5)

n = 0
for train_index, test_index in skf.split(data, target):
    start_time = timeit.default_timer()
    path_results_test = path_results+'/test' + str(n) + '/'
    if not os.path.exists(path_results_test):
        os.makedirs(path_results_test)
    train_set_x,train_set_y = data[train_index],target[train_index]
    test_set_x,test_set_y = data[test_index],target[test_index]
    clf = svm.LinearSVC()#svm.SVC(kernel='rbf', decision_function_shape='ovr',verbose=True)
    clf.fit(train_set_x, train_set_y)
    score = clf.score(test_set_x, test_set_y)
    print(score)
    end_time = timeit.default_timer()
    print('The code ' + ' ran for %.2fm' % ((end_time - start_time) / 60.))