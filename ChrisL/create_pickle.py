# '/Users/chris/Documents/GitHub/Stat440-Module3/Modified_data/garbage/tr'

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import pickle

train_dir = "/Users/chris/Documents/GitHub/Stat440-Module3/Modified_data/garbage/tr"
test_dir = "/Users/chris/Documents/GitHub/Stat440-Module3/Modified_data/garbage/te"
IMG_SIZE = 200

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

# Code taken from: https://pythonprogramming.net/loading-custom-data-deep-learning-python-tensorflow-keras/
def create_x_data(data,DATADIR):
	
	path = DATADIR  # create path 
	for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
		try:
			img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
			new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
			data.append([new_array])  # add this to our data
		except Exception as e:  # in the interest in keeping the output clean...
			pass
		#except OSError as e:
		#    print("OSErrroBad img most likely", e, os.path.join(path,img))
		#except Exception as e:
		#    print("general exception", e, os.path.join(path,img))

create_x_data(training_data,train_dir)
X_train = np.array(training_data).reshape(-1,IMG_SIZE,IMG_SIZE,1)
y_train = class_num

create_x_data(testing_data,test_dir)
X_test = np.array(testing_data).reshape(-1,IMG_SIZE,IMG_SIZE,1)


# save data
pickle_out = open("X_train.pickle","wb")
pickle.dump(X_train,pickle_out)
pickle_out.close()

pickle_out = open("y_train.pickle","wb")
pickle.dump(y_train,pickle_out)
pickle_out.close()

pickle_out = open("X_test.pickle","wb")
pickle.dump(X_test,pickle_out)
pickle_out.close()

