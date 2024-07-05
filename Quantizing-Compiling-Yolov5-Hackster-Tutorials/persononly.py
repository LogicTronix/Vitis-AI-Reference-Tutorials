import os
import shutil

labels_folder = 'datasets/bdd/val/labels_new'
labels_new = 'datasets/bdd/val/labels_person'

for filenames in os.listdir(labels_folder):
    with open(os.path.join(labels_folder,filenames), 'r') as source_file:
        lines = source_file.readlines()

    with open(os.path.join(labels_new, filenames), 'w') as destination_file:
        for line in lines:
            if line.startswith('0'):
                destination_file.write(line)


print("Process complete")