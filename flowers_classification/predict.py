import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from keras.preprocessing import image
from keras.applications.densenet import preprocess_input, decode_predictions
from keras.models import load_model


model = load_model('flower_model.h5')
flowers_path = Path('flowers/')

flower_types = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
test_size = 100
wrong_to_show = 3
wrong_pr = dict()

for species in flower_types:
	all_names = os.listdir(flowers_path / species)
	perm = np.random.permutation(all_names)
	wrong_pr[species] = []
	
	for name in perm[:test_size]:
		img_path = str(flowers_path / species / name)
		img = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)

		preds = (model.predict(x)[0]).tolist()
		ind = preds.index(max(preds))
		
		if flower_types[ind] != species:
			wrong_pr[species].append((img_path, flower_types[ind]))
	
	print(species, ': ', (100 - len(wrong_pr[species])) / (test_size / 100), '%')
	wrong_pr[species] = wrong_pr[species][:wrong_to_show]


fig = plt.figure(figsize=(8, 14))
columns = 3
rows = 5

for i in range(1, columns * rows + 1):
	a = (i - 1) % wrong_to_show
	b = (i - 1) // wrong_to_show
	
	wr_tup = wrong_pr[ flower_types[b] ]
	
	if a >= len(wr_tup):
		continue
	
	img_path = wr_tup[a][0]
	img = image.load_img(img_path, target_size=(224, 224))
	
	fig.add_subplot(rows, columns, i)
	plt.imshow(img)
	plt.title(wr_tup[a][1])
	plt.axis('off')

plt.show()