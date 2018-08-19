from keras.applications.densenet import DenseNet121
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten
from keras.models import Model
from keras.optimizers import Adam
import numpy as np

def get_model():
	base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(224,224,3))

	for layer in base_model.layers:
		layer.trainable = False

	base_model_ouput = base_model.output

	x = Flatten()(base_model.output)
	x = Dense(250, activation='relu', name='fc1')(x)
	x = Dropout(0.2)(x)
	x = Dense(5, activation='softmax', name='fc2')(x)

	return Model(inputs=base_model.input, outputs=x)


model = get_model()

opt = Adam(lr=1e-3, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


model.save('flower_model.h5')