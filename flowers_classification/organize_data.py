import os
import shutil
from pathlib import Path

import numpy as np

flowers_path = Path('flowers/')
valid_path = Path('./data/valid')
train_path = Path('./data/train')

flower_types = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
val_size = 50


for species in flower_types:
	all_names = os.listdir(flowers_path / species)
	perm = np.random.permutation(all_names)
	
	for name in perm[:val_size]:
		shutil.copyfile(str(flowers_path / species / name), str(valid_path / species / name))
	
	for name in perm[val_size:]:
		shutil.copyfile(str(flowers_path / species / name), str(train_path / species / name))