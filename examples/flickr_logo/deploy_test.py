'''
Script to test images using a trained model
'''

import numpy as np
import caffe
import cv2
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import math

#~ path_prefix = os.getcwd()
path_prefix = '/home/nikhil/git_repo/caffe/examples/flickr_logo'


'''
deployment test function
'''
def test():
    '''
    # dictionary mapping label strings to  to correspoding values
    labels = {
              0: 'dhl',
              1: 'fedex',
              2: 'cocacola',
              3: 'hp',
              4: 'ferrari',
              5: 'starbucks',
              6: 'ford',
              7: 'texaco',
              8: 'google',
              9: 'pepsi',
              10: 'bmw',
              11: 'adidas',
              12: 'apple',
              13: 'heineken'
    }

    # dictionary mapping keys to correspoding label strings
    rev_labels = {
              'dhl': 0,
              'fedex': 1,
              'cocacola': 2,
              'hp': 3,
              'ferrari': 4,
              'starbucks': 5,
              'ford': 6,
              'texaco': 7,
              'google': 8,
              'pepsi': 9,
              'bmw': 10,
              'adidas': 11,
              'apple': 12,
              'heineken': 13
    }
    '''

    labels = {
        0: 'ups', 1: 'dhl', 2: 'fedex', 3: 'fosters', 4: 'cocacola',
        5: 'hp', 6: 'erdinger', 7: 'ferrari', 8: 'carlsberg', 9: 'starbucks',
        10: 'corona', 11: 'tsingtao', 12: 'ford', 13: 'stellaarthois', 14: 'esso',
        15: 'google', 16: 'singha', 17: 'pepsi', 18: 'guiness', 19: 'bmw',
        20: 'texaco', 21: 'shell', 22: 'adidas', 23: 'rittersport', 24: 'becks',
        25: 'apple', 26: 'aldi', 27: 'nvidia', 28: 'milka',
        29: 'paulaner', 30: 'heineken', 31: 'chimay'
    }

    rev_labels = {
        'ups': 0, 'dhl': 1, 'fedex': 2, 'fosters': 3, 'cocacola': 4,
        'hp': 5, 'erdinger': 6, 'ferrari': 7, 'carlsberg': 8, 'starbucks': 9,
        'corona': 10, 'tsingtao': 11, 'ford': 12, 'stellaarthois': 13, 'esso': 14,
        'google': 15, 'singha': 16, 'pepsi': 17, 'guiness': 18, 'bmw': 19,
        'texaco': 20, 'shell': 21, 'adidas': 22, 'rittersport': 23, 'becks': 24,
        'apple': 25, 'aldi': 26, 'nvidia': 27, 'milka': 28,
        'paulaner': 29, 'heineken': 30, 'chimay': 31
    }


    # load neural network architecture modified for deployment version
    deploy = 'examples/flickr_logo/models/resnet-deploy.prototxt'

    # load trained model containing weights of features
    model = 'examples/flickr_logo/snapshot/snapshot_resnet_flickr_32_iter_40000.caffemodel'

    # file containing list of images to test
    img_list = 'examples/flickr_logo/data/test_32_path.txt'

    # load mean of the trained dataset
    mean_path = 'examples/flickr_logo/data/mean/flickr_logo_train_all_mean.binaryproto'

    # create a network with the specified parameters
    net = caffe.Net(deploy,
                    model,
                    caffe.TEST)

    # read mean file specified
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(mean_path, 'rb' ).read()
    blob.ParseFromString(data)
    arr = np.array( caffe.io.blobproto_to_array(blob) )
    mean = arr[0]

    # lists of predicted and real labels
    y_true = []
    y_pred = []

    # predict output label for each image
    with open(img_list, 'r') as fd:
        indexlist = fd.readlines()
        for line in indexlist:
            # get true label from the filename
            index = line.strip().split()
            img_path = index[0]
            label = index[1]
            fext = os.path.basename(img_path)
            fname = fext.strip().split('_')[0]
            y_true.append(rev_labels[fname])

            # read image
            im = cv2.imread(img_path, -1)
            # resize image
            im = cv2.resize(im, (256, 256), interpolation = 2) #256

            # subtract mean from image
            #~ im = np.float32(im)
            #~ im = im.transpose((2, 0, 1))
            #~ im -= mean

            #~ image_mean = [128, 128, 128]
            #~ channel_mean = np.zeros((3, 256, 256))
            #~ for channel_index, mean_val in enumerate(image_mean):
                #~ channel_mean[channel_index, ...] = mean_val
            #~ im -= channel_mean

            #~ im = im.transpose((1, 2, 0))

            '''
            test time augmentation
            predict labels for variations of the image
            average the results for all predictions
            '''
            # dictionary of probabilities for each image
            prob_dict = {k: [] for k in range(32)}
            avg_l = [-1]*32

            # get prdiction probabilites for all labels
            out = predict(im, net)
            prob_dict = update_dict(out, prob_dict)

            im_90 = rotate_image(im, -5)
            out = predict(im_90, net)
            prob_dict = update_dict(out, prob_dict)

            im_180 = rotate_image(im, 5)
            out = predict(im_180, net)
            prob_dict = update_dict(out, prob_dict)

            im_270 = rotate_image(im, 10)
            out = predict(im_270, net)
            prob_dict = update_dict(out, prob_dict)

            im_sc_90 = scale_image(im, 0.9)
            out = predict(im_sc_90, net)
            prob_dict = update_dict(out, prob_dict)

            im_sc_110 = scale_image(im, 1.1)
            out = predict(im_sc_110, net)
            prob_dict = update_dict(out, prob_dict)

            # average out the probabilities for each label
            for key, value in prob_dict.iteritems():
                if math.isnan(float(np.mean(value))):
                    avg_l[key] = float(0)
                else:
                    avg_l[key] = np.mean(value)

            # predicted label
            y_pred.append(np.asarray(avg_l).argmax())

            # print logs if image is misclassified
            if rev_labels[fname] != np.asarray(avg_l).argmax():
                print "Misclassified Image:", fext, "| Actual:", rev_labels[fname], "| Prediction:", np.asarray(avg_l).argmax()

                top_k = np.asarray(avg_l).argsort()[-3:][::-1]
                print "Top_K prediction for file", os.path.basename(img_path), "is", top_k
                for i in top_k:
                    print "label", i, labels[i], " | prob", float("{0:.3f}".format(avg_l[i]))
                print

    print "\nUnique Elements:"
    print "True", set(y_true)
    print "Pred", set(y_pred)
    print "\nConfusion matrix:"
    print confusion_matrix(y_true, y_pred)
    print "Accuracy", accuracy_score(y_true, y_pred)

'''
update the probabilities for each label in the dictionary
'''
def update_dict(out, prob_dict):
    out_l = out.tolist()
    for i in sorted(out_l):
        prob_dict[out_l.index(i)].append(i)

    return prob_dict

'''
get output from caffe for the given image
'''
def predict(im, net):
    # crop and tranpose image
    im = crop_image(im, 227, False)
    im = im.transpose((2, 0, 1))

    # send transformed image caffe network
    net.blobs['data'].reshape(1, 3, 227, 227)
    net.blobs['data'].data[...] = im
    out = net.forward()
    return out['prob'][0]

'''
rotate the image
'''
def rotate_image(im, rot_angle):
    rows, cols, ch = im.shape

    M = cv2.getRotationMatrix2D((cols/2, rows/2), rot_angle, 1)
    rot_image = cv2.warpAffine(im, M, (cols, rows))
    return rot_image

'''
scale the image by given scale factor
'''
def scale_image(im, scale):
    rows, cols, ch = im.shape
    scl_image = cv2.resize(im, None, fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)

    return scl_image

'''
crop image to given size
'''
def crop_image(im, crop_size, train):
    h = im.shape[0]
    w = im.shape[1]
    #~ h_crop = (h-crop_size)/2
    #~ w_crop = (w-crop_size)/2
    if train:
        h_start = random.randint(1, h-crop_size)
        w_start = random.randint(1, w-crop_size)
    else:
        h_start = (h-crop_size)/2
        w_start = (w-crop_size)/2
    h_end = h_start + crop_size
    w_end = w_start + crop_size

    crop_img = im[h_start:h_end, w_start:w_end] # img[y: y + h, x: x + w]
    return crop_img

'''
main function
'''
if __name__ == '__main__':
    test()
