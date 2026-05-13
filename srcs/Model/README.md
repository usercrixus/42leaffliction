# Model

`model.py` creates the ResNet-18 model and changes the last layer to match our
number of classes.

`loader.py` loads images from class folders, applies the same preprocessing to
all images, and splits the dataset into train and validation parts.

`train.py` trains the model, checks validation accuracy after each epoch, prints
the efficiency rate during training, and saves the best checkpoint.

`predict.py` loads a saved checkpoint, prepares one image the same way as
training, predicts its class, and displays the result.

## Summary

Training loads all images, then `loader.py` splits them into 80% train and 20%
validation.

Prediction does not use a split. `predict.py` only predicts the single image
path given in the command.
