# '/Users/chris/Documents/GitHub/Stat440-Module3/Modified_data/garbage/tr'

import numpy as np
import os
import pickle
from PIL import Image
import glob

train_dir = "/Users/chris/Documents/GitHub/Stat440-Module3/Modified_data/garbage/tr"
test_dir = "/Users/chris/Documents/GitHub/Stat440-Module3/Modified_data/garbage/te"
IMG_WIDTH = 384
IMG_LENGTH = 512

# Code taken from: https://stackoverflow.com/questions/57111243/how-to-read-multiple-text-files-in-a-folder-with-python
new_list = []
for root, dirs, files in os.walk("/Users/chris/Documents/GitHub/Stat440-Module3/Modified_data/garbage/tr_labels"):
	for file in sorted(files):
		with open(os.path.join(root, file), 'r') as f:
			text = f.read()
			new_list.append(text)
class_num = [x[0] for x in new_list]                



training_data = []
testing_data = []

# Code taken from: https://stackoverflow.com/questions/51178166/iterate-through-folder-with-pillow-image-open

images = []
for f in sorted(glob.glob("/Users/chris/Documents/GitHub/Stat440-Module3/Modified_data/garbage/tr/*")):
    images.append(np.asarray(Image.open(f)))

train_images = np.array(images)
train_labels = np.array(class_num,dtype = "uint8").reshape(-1,1)


images_test = []
for ft in sorted(glob.glob("/Users/chris/Documents/GitHub/Stat440-Module3/Modified_data/garbage/te/*")):
    images_test.append(np.asarray(Image.open(ft)))

test_images = np.array(images_test)


validation_images, validation_labels = train_images[:100], train_labels[:100]
train_images, train_labels = train_images[100:], train_labels[100:]
print(validation_images.shape)
print(train_images.shape)

# # save data
pickle_out = open("train_images.pickle","wb")
pickle.dump(train_images,pickle_out)
pickle_out.close()

pickle_out = open("train_labels.pickle","wb")
pickle.dump(train_labels,pickle_out)
pickle_out.close()

pickle_out = open("validation_images.pickle","wb")
pickle.dump(validation_images,pickle_out)
pickle_out.close()

pickle_out = open("validation_labels.pickle","wb")
pickle.dump(validation_labels,pickle_out)
pickle_out.close()

pickle_out = open("test_images.pickle","wb")
pickle.dump(test_images,pickle_out)
pickle_out.close()

print("Successfully Read In Images!")
