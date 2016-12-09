from glob import glob
import os
import random
import sys
from sklearn.model_selection import train_test_split
#~ import sklearn

#~ path_prefix = os.getcwd()
path_prefix = '/home/nikhil/git_repo/caffe/examples/flickr_logo'

# path of the directory containing images
#~ img_path = path_prefix + '/data/merged_resized/'
img_path = path_prefix + '/data/train/'

# read jpeg images from this directory
img_list = glob(img_path + '*.jpg')

# dictionary of values associated to label names
labels = {
 'dhl': 0, 'fedex': 1, 'starbucks': 2, 'redbull': 3, 'vodafone': 4,
 'google': 5, 'ferrari': 6, 'unicef': 7, 'bmw': 8, 'sprite': 9,
 'porsche': 10, 'yahoo': 11, 'hp': 12, 'puma': 13, 'adidas': 14,
 'nbc': 15,'cocacola': 16, 'texaco': 17, 'citroen': 18, 'heineken': 19,
 'apple': 20, 'nike': 21, 'mini': 22, 'ford': 23, 'pepsi': 24,
 'mcdonalds': 25, 'intel': 26, 'ups': 27, 'HP': 28, 'fosters': 29,
 'erdinger': 30, 'carlsberg': 31, 'corona': 32, 'tsingtao': 33, 'stellaarthois': 34
 'esso': 35, 'singha': 36, 'guiness': 37, 'shell': 38, 'rittersport': 39,
 'becks': 40, 'milka': 41, 'aldi': 42, 'nvidia': 43, 'paulaner': 44, 'chimay': 45
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
random.shuffle(img_list)

Y = [0] * len(img_list)
print "len(img_list) ", len(img_list)

X_train, X_test, Y_train, Y_test = train_test_split(img_list, Y, test_size=0.2, random_state=42)
print "X_train ", len(X_train)
print "Y_train ", len(Y_train)
print "X_test ", len(X_test)
print "Y_test ", len(Y_test)

orig_stdout = sys.stdout
data_path_train = path_prefix + '/data/merged_train_path.txt'

f = file(data_path_train, 'w')
sys.stdout = f

for f_path in X_train:
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

data_path_test = path_prefix + '/data/merged_test_path.txt'
f = file(data_path_test, 'w')
sys.stdout = f

for f_path in X_test:
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

#~ data_path_train = path_prefix + '/data/merged_train_path.txt'
data_path_train = path_prefix + '/data/merged_train_filenames.txt'

f = file(data_path_train, 'w')
sys.stdout = f

for f_path in X_train:
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

#~ data_path_test = path_prefix + '/data/merged_test_path.txt'
data_path_test = path_prefix + '/data/merged_test_filenames.txt'

f = file(data_path_test, 'w')
sys.stdout = f

for f_path in X_test:
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
