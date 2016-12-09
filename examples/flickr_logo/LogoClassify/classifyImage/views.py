from django.shortcuts import render,redirect, render_to_response
from django.http import HttpResponse,JsonResponse
from django.views.decorators.csrf import csrf_exempt
import hashlib # For SHA-256 Encoding
import base64
from base64 import b64decode
import json
import os
import uuid
import json
import urllib2
import urllib
import numpy as np
import cv2
import math
import wget
import caffe
from LogoClassify.settings import MEDIA_ROOT
from tf import run_inference_on_image
from tf import maybe_download_and_extract

path_prefix = '/home/nikhil/git_repo/caffe/examples/flickr_logo/' # TODO
neural_net = None

'''
load a new with the given architecture and weights of a trained
caffe model
'''
def load_network():
    # load neural network architecture modified for deployment version
    deploy = path_prefix + 'models/resnet-deploy.prototxt'

    # load trained model containing weights of features
    model = path_prefix + 'snapshot/snapshot_resnet_flickr_32_iter_40000.caffemodel'

    # create a network with the specified parameters
    global neural_net
    neural_net = caffe.Net(deploy,
                    model,
                    caffe.TEST)

'''
deployment test function
'''
def test(img_list):
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

    # load mean of the trained dataset
    mean_path = path_prefix + 'data/mean/flickr_logo_train_all_mean.binaryproto'

    # read mean file specified
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(mean_path, 'rb' ).read()
    blob.ParseFromString(data)
    arr = np.array( caffe.io.blobproto_to_array(blob) )
    mean = arr[0]

    for img_path in img_list:
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
	out = predict(im, neural_net)
	prob_dict = update_dict(out, prob_dict)

	im_90 = rotate_image(im, -5)
	out = predict(im_90, neural_net)
	prob_dict = update_dict(out, prob_dict)

	im_180 = rotate_image(im, 5)
	out = predict(im_180, neural_net)
	prob_dict = update_dict(out, prob_dict)

	im_270 = rotate_image(im, 10)
	out = predict(im_270, neural_net)
	prob_dict = update_dict(out, prob_dict)

	im_sc_90 = scale_image(im, 0.9)
	out = predict(im_sc_90, neural_net)
	prob_dict = update_dict(out, prob_dict)

	im_sc_110 = scale_image(im, 1.1)
	out = predict(im_sc_110, neural_net)
	prob_dict = update_dict(out, prob_dict)

	# average out the probabilities for each label
	for key, value in prob_dict.iteritems():
	    if math.isnan(float(np.mean(value))):
		avg_l[key] = float(0)
	    else:
		avg_l[key] = np.mean(value)

	# predicted label
	y_pred = np.asarray(avg_l).argmax()

	print "Predicted Label: ", labels[y_pred]
	return labels[y_pred]


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
    global neural_net
    neural_net.blobs['data'].reshape(1, 3, 227, 227)
    neural_net.blobs['data'].data[...] = im
    out = neural_net.forward()
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
create views here
'''
def save_file(f):
    original_name, file_extension = os.path.splitext(f.name)
    filename =  uuid.uuid4().hex + file_extension
    path =  os.path.join(MEDIA_ROOT, filename)
    destination = open(path, 'wb+')
    for chunk in f.chunks():
        destination.write(chunk)
    destination.close()
    return path

'''
load network initially
'''
def index(request):
    load_network()
    maybe_download_and_extract()
    return  render_to_response('index.html')

'''
called when 'Upload Image' is clicked
'''
@csrf_exempt
def uploadImage(request):
    print request.FILES
    if request.FILES and request.FILES.get('file_upload'):
        path = save_file( request.FILES.get('file_upload') )
        tmp_list = []
        tmp_list.append(path)
        res = test(tmp_list)
        tf_info = run_inference_on_image(path)
        returnObject = {}
        returnObject['logo'] = res
        returnObject['otherInfo'] = tf_info
    resp = json.dumps( returnObject )
    return HttpResponse(resp);

'''
called when 'Submit URL' is clicked
'''
@csrf_exempt
def submitUrl(request):
    url = request.POST.get( 'url' )
    original_name, file_extension = os.path.splitext(url)
    filename =  uuid.uuid4().hex + file_extension
    path =  os.path.join(MEDIA_ROOT, filename)

    resp = urllib.urlopen(url)
    img = np.asarray(bytearray(resp.read()), dtype="uint8")
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    cv2.imwrite(path, img)

    tmp_list = []
    tmp_list.append(path)
    res = test(tmp_list)
    tf_info = run_inference_on_image(path)
    returnObject = {}
    returnObject['logo'] = res
    returnObject['otherInfo'] = tf_info
    resp = json.dumps( returnObject )
    return HttpResponse(resp);
