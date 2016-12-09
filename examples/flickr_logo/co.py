#~ from re import split
import os
import shutil

logos = {}
image_list = []

with open("flickr_logos_27_dataset_training_set_annotation.txt", "r") as data:
    for line in data:
	key = line.decode('utf-8').strip().split()

	file_from = "flickr_logos_27_dataset_images/" + key[0]
	if (key[0] not in image_list):
	    image_list.append(key[0])
	    key[1] = key[1].lower()

	    if key[1] in logos:
		logos[key[1]] += 1
	    else:
		logos[key[1]] = 1

	    file_to = "all/" + key[1] + "_" + str(logos[key[1]]) + "." + key[0].split('.')[1]
	    print "Copied ", key[0].split('.')[1], " to ", file_to
	    shutil.copy2(file_from, file_to)
