from glob import glob
import os

#~ img_dir = '/home/nikhil/git_repo/caffe/examples/flickr_logo/data/flickr_32_all/'
img_dir = '/home/nikhil/git_repo/caffe/examples/flickr_logo/data/train_32_resized/'

#~ img_dir1 = '/home/nikhil/git_repo/caffe/examples/flickr_logo/data/flickr_27_all/'

img_list = glob(img_dir + '*.jpg')
#~ img_list1 = glob(img_dir1 + '*.jpg')

logos = {}

k = 0
for i in img_list:
    f_base = os.path.basename(i)
    fname = f_base.strip().split('_')[0]
    if fname not in logos:
        logos[fname] = k
        k += 1

#~ for i in img_list1:
    #~ f_base = os.path.basename(i)
    #~ fname = f_base.strip().split('_')[0]
    #~ if fname not in logos:
        #~ logos[fname] = k
        #~ k += 1

print "logos"
print logos

#~ {
#~ 'dhl': 0, 'fedex': 1, 'starbucks': 2, 'redbull': 3, 'vodafone': 4,
 #~ 'google': 5, 'ferrari': 6, 'unicef': 7, 'bmw': 8, 'sprite': 9,
 #~ 'porsche': 10, 'yahoo': 11, 'hp': 12, 'puma': 13, 'adidas': 14,
 #~ 'nbc': 15,'cocacola': 16, 'texaco': 17, 'citroen': 18, 'heineken': 19,
 #~ 'apple': 20, 'nike': 21, 'mini': 22, 'ford': 23, 'pepsi': 24,
 #~ 'mcdonalds': 25, 'intel': 26, 'ups': 27, 'HP': 28, 'fosters': 29,
 #~ 'erdinger': 30, 'carlsberg': 31, 'corona': 32, 'tsingtao': 33, 'stellaarthois': 34
 #~ 'esso': 35, 'singha': 36, 'guiness': 37, 'shell': 38, 'rittersport': 39,
 #~ 'becks': 40, 'milka': 41, 'aldi': 42, 'nvidia': 43, 'paulaner': 44, 'chimay': 45
#~ }

{
 'google': 15, 'apple': 25, 'adidas': 22, 'paulaner': 29, 'guiness': 18,
 'singha': 16, 'cocacola': 4, 'dhl': 1, 'texaco': 20, 'fosters': 3,
 'fedex': 2, 'aldi': 26, 'chimay': 31, 'shell': 21, 'stellaarthois': 13,
 'becks': 24, 'hp': 5, 'tsingtao': 11, 'ford': 12, 'carlsberg': 8,
 'starbucks': 9, 'pepsi': 17, 'esso': 14, 'heineken': 30,
 'erdinger': 6, 'corona': 10, 'milka': 28, 'ferrari': 7, 'nvidia': 27,
 'rittersport': 23, 'ups': 0, 'bmw': 19
}
