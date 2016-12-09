from glob import glob
import os
import random
import sys
import argparse
from sklearn.model_selection import train_test_split

#~ path_prefix = os.getcwd()
path_prefix = '/home/nikhil/git_repo/caffe/examples/flickr_logo'

# path of the directory containing images
img_path = path_prefix + '/data/merged_resized/'


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    argparser.add_argument("--dirpath", help="Directory path of images\n",
                           type=str, required=True)

    args = argparser.parse_args()
    img_path = args.dirpath

    # read jpeg images from this directory
    img_list = glob(img_path + '*.jpg')

    # shuffle image list
    random.shuffle(img_list)

    Y = [0] * len(img_list)

    X_train, X_test, Y_train, Y_test = train_test_split(img_list, Y, test_size=0.2, random_state=42)
    print img_path
    print "X_train ", len(X_train)
    print "Y_train ", len(Y_train)
    print "X_test ", len(X_test)
    print "Y_test ", len(Y_test), "\n"

    if not os.path.exists(img_path+'train'):
        os.makedirs(img_path+'train')

    if not os.path.exists(img_path+'test'):
        os.makedirs(img_path+'test')

    for i in X_train:
        os.rename(i, img_path+'train/'+i.split('/')[-1])

    for i in X_test:
        os.rename(i, img_path+'test/'+i.split('/')[-1])
