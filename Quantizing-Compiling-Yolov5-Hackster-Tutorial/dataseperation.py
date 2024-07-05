import os
import shutil

labels_folder = 'datasets/bdd/test/labels'
images_folder = 'datasets/bdd/test/images'

new_images = 'datasets/bdd/test/images_new'
new_labels = 'datasets/bdd/test/labels_new'

person_labels = []

# create images folder if it doesn't exist
os.makedirs(images_folder, exist_ok=True)

# iterate over files in "labels" folder
for label_filename in os.listdir(labels_folder):
    if label_filename.endswith(".txt"):
        # Load the content of the label file
        with open(os.path.join(labels_folder, label_filename), 'r') as label_file:
            lines = label_file.readlines()

        # Check if the label file contains "person" i.e. 0
        for line in lines:
            # print(type(line[0]))

            if line[0] == '0':
                # print(line[0])
                # print("found pedestrian")
                # Copy the label file to the "images" folder
                shutil.copy(os.path.join(labels_folder, label_filename), os.path.join(new_labels, label_filename))

                # Generate the image file name
                image_filename = os.path.splitext(label_filename)[0] + ".jpg"

                # Check if the corresponding image file exists
                if os.path.exists(os.path.join(images_folder, image_filename)):
                    # Copy the image file to the "images" folder
                    shutil.copy(os.path.join(images_folder, image_filename), os.path.join(new_images, image_filename))
            continue

print("Process completed.")

