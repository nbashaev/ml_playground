Baseline for [a kaggle flower dataset](https://www.kaggle.com/alxmamaev/flowers-recognition) with using Keras. The approach is to apply transfer learning technique to DenseNet121 pretrained on ImageNet

If you want to try it on your own, then:

- copy all folders with pictures from the dataset into the directory "flowers"
- run `organize_data.py` and `create_model.py`
- run `train_model.py [num of epochs]`
- to see which pictures were wrongly classified run `predict.py`

Notice that during the training only the models with the best accuracy are saved. And don't forget to rename the proper checkpoint into `flower_model.h5` before running `train_model.py` or `predict.py`
