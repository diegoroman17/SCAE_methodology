import numpy
import gzip
import pickle as cPickle
#import matplotlib.pyplot as plt
from sklearn.metrics import *

def save2file(filename, data):
    f = gzip.open(filename,'wb')
    cPickle.dump(data, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    
    
def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar

def training_curves(path_in, path_out):
    f = gzip.open(path_in, 'rb')
    (train_cost, val_cost, test_cost) = cPickle.load(f)
    f.close()
    t0, t1, t2 = zip(*train_cost)
    v0, v1, v2 = zip(*val_cost)
    te0, te1, te2 = zip(*test_cost)
    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    ax.plot(v1,v2,'g',te1,te2,'b')
    fig.savefig(path_out,format='pdf')   # save the figure to file
    plt.close(fig)    # close the figure

def metrics(path_in, path_out):
    f = gzip.open(path_in, 'rb')
    v = cPickle.load(f)
    f.close()
    y_pred = []
    y = []
    for i in range(len(v)):
        y_pred.extend(v[i][0])
        y.extend(v[i][1])
    with open(path_out, 'w') as f:    
        f.write('Confusion matrix:\n' + str(confusion_matrix(y, y_pred))+'\n\n')
        f.write('Accuracy score:\n' + str(accuracy_score(y, y_pred)) + '\n\n')
        f.write('Classification report:\n'+str(classification_report(y, y_pred))+'\n\n')
        f.write('Hamming loss:\n'+str(hamming_loss(y, y_pred))+'\n\n')
        f.write('Jaccard similarity score:\n'+str(jaccard_similarity_score(y, y_pred))+'\n\n')     
        f.write('Zero one loss:\n'+str(zero_one_loss(y, y_pred))+'\n\n')
    
    
def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = numpy.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = numpy.zeros(out_shape, dtype=dt)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array

    
def main():
    metrics('C:\\Users\\User\\Dropbox\\Seville\\data\\kfolds\\resultsSCAE3\\test4\\test_result.pkl.gz',
            'C:\\Users\\User\\Dropbox\\Seville\\data\\kfolds\\resultsSCAE3\\test4\\test.txt')
    
if __name__ == '__main__':
    main()
