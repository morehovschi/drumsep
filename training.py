"""
Marius Orehovschi
Jan 2021
Convolutional neural network for drum signal separation.

This file contains the network training process.
"""

import numpy as np
import os
import json

import tensorflow
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import LeakyReLU

# print out info about available GPUs
print("Num GPUs Available: ", len(tensorflow.config.experimental.list_physical_devices('GPU')))

class My_Custom_Generator(tensorflow.keras.utils.Sequence) :
	"""
		Training/validation data generator. Loads and returns (batch_size) framed STFT images and their 
		corresponding ideal binary masks.
	"""
  
	def __init__(self, fname_pairs, batch_size) :
		self.fname_pairs = fname_pairs
		self.batch_size = batch_size


	def __len__(self) :
		return (np.ceil(len(self.fname_pairs) / float(self.batch_size))).astype(np.int)


	def __getitem__(self, idx):

		# shuffle all the data and then use the first (batch_size) items
		np.random.shuffle(self.fname_pairs)
		batch_fnames = self.fname_pairs[:batch_size]
		batch_x = batch_fnames[:, 0]
		batch_y = batch_fnames[:, 1]

		returnable_x = np.expand_dims(np.array([np.load(str(file_name)) for file_name in batch_x]), axis=-1)
		returnable_y = np.array([np.load(str(file_name)) for file_name in batch_y])

		return returnable_x, returnable_y

fname_pairs = np.load('/storage/moreho21/musdb18hq/preprocessed_data/fnames.npy')

# shuffle the data then split into train and validation
np.random.shuffle(fname_pairs)
val_train_split = 0.2
val_fnames = fname_pairs[:int(val_train_split*len(fname_pairs))]
train_fnames = fname_pairs[int(val_train_split*len(fname_pairs)):]

print("Val size:", len(val_fnames), "Train size:", len(train_fnames))

batch_size = 512

training_generator = My_Custom_Generator(train_fnames, batch_size)
validation_generator = My_Custom_Generator(val_fnames, batch_size)

train_size = len(train_fnames)
val_size = len(val_fnames)

prop_dataset = .1 # propotion of dataset to be used in 1 epoch
steps_per_epoch = int(train_size * prop_dataset / batch_size)
validation_steps = int(val_size * prop_dataset / batch_size)

# define the model
model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', input_shape=(513, 25, 1)))
model.add(LeakyReLU())
model.add(Conv2D(16, (3,3), padding='same'))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3,3), padding='same'))
model.add(LeakyReLU())
model.add(Conv2D(16, (3,3), padding='same'))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
model.add(LeakyReLU())
model.add(Dropout(0.5))
model.add(Dense(513, activation='sigmoid'))
precision = tensorflow.keras.metrics.Precision()
recall = tensorflow.keras.metrics.Recall()
model.compile(loss=tensorflow.keras.losses.BinaryCrossentropy(), optimizer='adam',
			  metrics=[precision, recall])

model_path = 'training_process/trained_model/'
model_name = "drumsep_full"

# container where all the smaller history objects will be saved
general_history = {
	"loss": [],
	"precision": [],
	"recall": [],
	"val_loss": [],
	"val_precision": [],
	"val_recall": [],
}

# fit 12 iterations of 4 epochs (broken down this way for frequent saving of the weights)
for i in range(12):
	history = model.fit_generator(generator=training_generator,
				   steps_per_epoch = steps_per_epoch,
				   epochs = 4,
				   verbose = 1,
				   validation_data = validation_generator,
				   validation_steps = validation_steps)

	general_history["loss"] += history.history["loss"]
	general_history["precision"] += history.history["precision"]
	general_history["recall"] += history.history["recall"]
	general_history["val_loss"] += history.history["val_loss"]
	general_history["val_precision"] += history.history["val_precision"]
	general_history["val_recall"] += history.history["val_recall"]

	model_json = model.to_json()
	with open(model_path + model_name + ".json", "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights(model_path + model_name + ".h5")
	print("Saved model %d to disk."% (i+1))

	for key in general_history.keys():
		np.save("training_process/history/" + key, general_history[key])

