from keras.applications.densenet import DenseNet121
from keras.preprocessing import image
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, Callback
import numpy as np
import sys


class TrainHistory(Callback):
	def on_train_begin(self, logs={}):
		self.acc = []
		self.val_acc = []

	def on_epoch_end(self, batch, logs={}):
		self.acc.append(logs.get('acc'))
		self.val_acc.append(logs.get('val_acc'))
	
	def __str__(self):
		inf = [' '.join([str(num) for num in metric]) for metric in [self.acc, self.val_acc]]
		return '\n'.join(inf)


batch_size = 8
epoch_num = int(sys.argv[1])

train_datagen = image.ImageDataGenerator(
	rescale=1./255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True)

test_datagen = image.ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
	'data/train',
	target_size=(224,224),
	batch_size=batch_size,
	class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
	'data/valid',
	target_size=(224,224),
	batch_size=batch_size,
	class_mode='categorical')

filepath = 'flower_model_{epoch:02d}_{val_acc:.2f}.h5'
model = load_model('flower_model.h5')
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)
history = TrainHistory()
callbacks_list = [checkpoint, history]

model.fit_generator(
	train_generator,
	steps_per_epoch = 4073 // batch_size,
	epochs = epoch_num,
	validation_data = validation_generator,
	validation_steps = 250 // batch_size,
	callbacks = callbacks_list)

with open('log.txt', 'a') as flog:
        flog.write(history)