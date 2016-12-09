'''
Python Data Layer for Caffe
'''

import numpy as np
import skimage.io
import caffe
import cv2
import random

'''
# Caffe layer
In 'python_param' parameter of 'python' layer:
    - module: the filename of this script
              'data_aug' in this case
    - layer: the name of the class
             'DataAuglayer' in this case
'''
class DataAugLayer(caffe.Layer):
    def setup(self, bottom, top):
        self.top_names = ['data', 'label']

        # read options set 'in param_str'
        params = eval(self.param_str)
        check_params(params)

        # store input as class variables
        self.batch_size = params['batch_size']

        # Create a batch loader to load the images.
        self.batch_loader = BatchLoader(params)

        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.
        top[0].reshape(self.batch_size, 3, 227, 227)
        top[1].reshape(self.batch_size, 1)


    def forward(self, bottom, top):
        for itt in range(self.batch_size):
            # Use the batch loader to load the next image.
            im, label = self.batch_loader.load_next_image()

            # Add directly to the caffe data layer
            top[0].data[itt, ...] = im
            top[1].data[itt, ...] = label

    def reshape(self, bottom, top):
        pass

    def backward(self, top, propagate_down, bottom):
        pass

'''
Class to load images in a separate thread and write to queue
I/O operation will run in paralell and does not wait on Caffe
'''
class BatchLoader(object):
    def __init__(self, params):
        # number of images per batch
        self.batch_size = params['batch_size']

        # read file containing image list
        fd = open(params['source'], 'r')
        self.indexlist = fd.readlines()
        fd.close()

        # counter for images read
        self._cur = 0

        # class instance for image preprocessing and transformations
        self.transformer = ImageAugTransformer(params)

        print "File list contains {} images".format(len(self.indexlist))

    def load_next_image(self):
        # Reset if one epoch is finished
        if self._cur == len(self.indexlist):
            self._cur = 0
            random.shuffle(self.indexlist)

        # Load an image
        index = self.indexlist[self._cur].strip().split()  # Get the image index
        file_path = index[0]
        label = index[1]

        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        self._cur += 1

        return self.transformer.preprocess(img), label

'''
Class to preprocess and transform images
'''
class ImageAugTransformer:
    def __init__(self, params):
        # read mean file specified
        data = open(params['mean_file'], 'rb').read()
        blob = caffe.proto.caffe_pb2.BlobProto()
        blob.ParseFromString(data)
        mean_arr = np.array(caffe.io.blobproto_to_array(blob))
        self.mean = mean_arr[0]

        # transpose values to convert image to BGR
        self.transpose = (2, 0, 1)

        # read image resizing parameters
        self.new_height = params['new_height']
        self.new_width = params['new_width']
        self.crop_size = params['crop_size']

        # train or test phase
        self.ph = params['ph']

    def preprocess(self, im):
        im = np.float32(im)

        # resize image if needed
        if (self.new_height > 0 or self.new_width > 0) and (self.new_height != im.shape[0] or self.new_width != im.shape[1]):
            im = cv2.resize(im, (self.new_width, self.new_height), interpolation = cv2.INTER_CUBIC)

        # subtract mean of dataset from each image
        #~ im = im.transpose((2, 0, 1))
        #~ im -= self.mean

        #~ image_mean = [128, 128, 128]
        #~ channel_mean = np.zeros((3, 256, 256))
        #~ for channel_index, mean_val in enumerate(image_mean):
            #~ channel_mean[channel_index, ...] = mean_val
        #~ im -= channel_mean

        #~ im = im.transpose((1, 2, 0))

        # data augmentation in training phase
        if self.ph is "train":
            im = scale_image(im)

            if bool(random.getrandbits(1)) is True:
                im = flip_image(im)

            im = crop_image(im, self.crop_size, True)

            im = rotate_image(im)

            if bool(random.getrandbits(1)) is True:
                im = change_image_brightness(im)

            #~ im = jitter_image_color(im)
        else:
            im = crop_image(im, self.crop_size, False)

        im = im.transpose((2, 0, 1))

        return im

'''
mirror of image
'''
def flip_image(im):
    flip_img = cv2.flip(im, 1)

    return np.float32(flip_img)

'''
rotate the image
'''
def rotate_image(im):
    rows, cols, ch = im.shape
    rotation_angles = [90, 180, 270]

    M = cv2.getRotationMatrix2D((cols/2, rows/2), random.randint(1, 360), 1) #random.choice(rotation_angles), 1)
    rot_image = cv2.warpAffine(im, M, (cols, rows))

    return np.float32(rot_image)

'''
scale the image randomly between 0.9 and 1.1
'''
def scale_image(im):
    rows, cols, ch = im.shape
    scaling_list = [0.9, 1.0, 1.1, 1.2]
    #~ scale = random.choice(scaling_list)
    scale = random.uniform(0.9, 1.1)
    scl_image = cv2.resize(im, None, fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)

    return np.float32(scl_image)

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
    return np.float32(crop_img)

'''
vary brightness/contrast of image
'''
def change_image_brightness(im):
    maxIntensity = 255.0

    if bool(random.getrandbits(1)):
        #increase brightness
        ch_img = maxIntensity*(im/maxIntensity)**0.5
        ch_img = np.array(ch_img, dtype='uint8')
    else:
        # increase contrast
        ch_img = maxIntensity*(im/maxIntensity)**2
        ch_img = np.array(ch_img, dtype='uint8')

    return np.float32(ch_img)

'''
adding noise to image
'''
def jitter_image_color(im):
    r_j = random.randint(-5, 5)
    g_j = random.randint(-5, 5)
    b_j = random.randint(-5, 5)

    R = im[:,:,0]
    G = im[:,:,1]
    B = im[:,:,2]

    jitter_img = np.dstack( (
        np.roll(R, r_j, axis=0), 
        np.roll(G, g_j, axis=1), 
        np.roll(B, b_j, axis=0)
        ))
    return np.float32(jitter_img)

'''
specify mandatory parameters to be entered
'''
def check_params(params):
    required = ['batch_size']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)
