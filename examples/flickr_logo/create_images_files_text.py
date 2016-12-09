from glob import glob
import os
import random
import sys

#~ path_prefix = os.getcwd()
path_prefix = '/home/nikhil/git_repo/caffe/examples/flickr_logo' # TODO

# path of the directory containing images
img_path1 = path_prefix + '/data/train_32_resized/'
img_path2 = path_prefix + '/data/test_32_resized/'

# read jpeg images from this directory
img_list_train = glob(img_path1 + '*.jpg')
img_list_test = glob(img_path2 + '*.jpg')

# dictionary of values associated to label names
labels = {
    'ups': 0, 'dhl': 1, 'fedex': 2, 'fosters': 3, 'cocacola': 4,
    'hp': 5, 'erdinger': 6, 'ferrari': 7, 'carlsberg': 8, 'starbucks': 9,
    'corona': 10, 'tsingtao': 11, 'ford': 12, 'stellaarthois': 13, 'esso': 14,
    'google': 15, 'singha': 16, 'pepsi': 17, 'guiness': 18, 'bmw': 19,
    'texaco': 20, 'shell': 21, 'adidas': 22, 'rittersport': 23, 'becks': 24,
    'apple': 25, 'aldi': 26, 'nvidia': 27, 'milka': 28,
    'paulaner': 29, 'heineken': 30, 'chimay': 31
}

#~ labels = {
 #~ 'dhl': 0,
 #~ 'fedex': 1,
 #~ 'cocacola': 2,
 #~ 'hp': 3,
 #~ 'ferrari': 4,
 #~ 'starbucks': 5,
 #~ 'ford': 6,
 #~ 'texaco': 7,
 #~ 'google': 8,
 #~ 'pepsi': 9,
 #~ 'bmw': 10,
 #~ 'adidas': 11,
 #~ 'apple': 12,
 #~ 'heineken': 13
#~ }

# shuffle image list
random.shuffle(img_list_train)
random.shuffle(img_list_test)

orig_stdout = sys.stdout
data_path_train = path_prefix + '/data/train_32_path.txt'

f = file(data_path_train, 'w')
sys.stdout = f

for f_path in img_list_train:
    # get name of the file ignoring its path
    f_base = os.path.basename(f_path)
    fname = f_base.strip().split('_')[0]

    if fname not in labels:
        print
        print "*******************************************************"
        print "Label not present in list ", fname
        print
        sys.exit(0)
        labels[fname] = len(labels)
    #~ print f_base, labels[fname]
    print f_path, labels[fname]

f.close

data_path_test = path_prefix + '/data/test_32_path.txt'
f = file(data_path_test, 'w')
sys.stdout = f

for f_path in img_list_test:
    # get name of the file ignoring its path
    f_base = os.path.basename(f_path)
    fname = f_base.strip().split('_')[0]

    if fname not in labels:
        print
        print "*******************************************************"
        print "Label not present in list ", fname
        print
        sys.exit(0)
        labels[fname] = len(labels)
    #~ print f_base, labels[fname]
    print f_path, labels[fname]

#~ data_path_train = path_prefix + '/data/train_32_path.txt'
data_path_train = path_prefix + '/data/train_32_filenames.txt'

f = file(data_path_train, 'w')
sys.stdout = f

for f_path in img_list_train:
    # get name of the file ignoring its path
    f_base = os.path.basename(f_path)
    fname = f_base.strip().split('_')[0]

    if fname not in labels:
        print
        print "*******************************************************"
        print "Label not present in list ", fname
        print
        sys.exit(0)
        labels[fname] = len(labels)
    print f_base, labels[fname]
    #~ print f_path, labels[fname]

f.close

#~ data_path_test = path_prefix + '/data/test_32_path.txt'
data_path_test = path_prefix + '/data/test_32_filenames.txt'

f = file(data_path_test, 'w')
sys.stdout = f

for f_path in img_list_test:
    # get name of the file ignoring its path
    f_base = os.path.basename(f_path)
    fname = f_base.strip().split('_')[0]

    if fname not in labels:
        print
        print "*******************************************************"
        print "Label not present in list ", fname
        print
        sys.exit(0)
        labels[fname] = len(labels)
    print f_base, labels[fname]
    #~ print f_path, labels[fname]

orig_stdout = sys.stdout
f.close
