'''
Resize images with OpenCV
'''

from glob import glob
import os
import cv2

resize_to = 512

path_prefix = '/home/nikhil/git_repo/caffe/examples/flickr_logo'
img_dir = path_prefix + '/data/test_32/'
dest_dir = path_prefix + '/data/test_32_resized/'

img_list = glob(img_dir + '*.jpg')

k = 0
for i in img_list:
    img = cv2.imread(i, -1) # cv2.IMREAD_COLOR 1, cv2.IMREAD_GRAYSCALE 0, cv2.IMREAD_UNCHANGED -1

    if k is not 0 and k%500 is 0:
        print "Processed", k, "images"

    if k is not 0 and k%(len(img_list)-1) is 0:
        print "Completed processing", len(img_list), "images"

    res_img = cv2.resize(img, (resize_to, resize_to), interpolation = 2)
    cv2.imwrite(dest_dir + os.path.basename(i).strip().split('.')[0] + ".jpg", res_img)
    k += 1
