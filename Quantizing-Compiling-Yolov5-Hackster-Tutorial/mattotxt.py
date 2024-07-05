import scipy.io
import numpy as np
import os
import shutil

folder_name = ''

for filename in os.listdir(folder_name):
    if filename.endswith('.mat'):
        rawname = os.path.splitext(filename)[0]
        txt_name = rawname + '.txt'
        # load the mat file
        mat = scipy.io.loadmat(filename)
        # convert matfile to txt format
        data = mat['']
        # saving the data to .txt format
        np.savetxt(txt_name, data, fmt='%f', delimiter=' ')

print("first part completed")

image_folder = ''
labels_folder = ''
annotation_folder = ''

os.makedirs(image_folder, exist_ok=True)
os.makedirs(labels_folder, exist_ok=True)
os.makedirs(annotation_folder, exist_ok=True)

for filename in os.listdir(folder_name):
    if filename.endswith('.mat'):
        shutil.copy(filename, os.path.join(annotation_folder, filename))
    elif filename.endswith('.txt'):
        shutil.copy(filename, os.path.join(labels_folder, filename))
    elif filename.endswith('.jpg'):
        shutil.copy(filename, os.path.join(image_folder, filename))
    else:
        print("NO specific file found")

print("Second part completed")