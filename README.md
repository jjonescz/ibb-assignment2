# IBB Assignment 2

This repository contains source code of my solution of Assignment 2 for Image
Based Biometry course at University of Ljubljana.

## Requirements

Python 3.8.2 was used with the following packages installed:

```txt
numpy==1.18.5
matplotlib==3.3.2
tensorflow==2.3.1
tensorflow-addons==0.11.2
```

Additionally, `data/` folder must contain unzipped dataset AWE-W provided in the
assignment (unzipped so that e.g. `data/train/0001.png` is valid path).

## Running

Script `model.py` can be run as-is. It will download EfficientNet-B0 weights and
use weights for the rest of the model saved in `out/final/weights.h5` to
construct the CNN model. It won't perform any training, just evaluation.

To train the model, change `TRAIN` constant defined near top of `model.py` to:

```py
TRAIN = True
```
