# Deepfake Detection with Deep Learning
## Marten Thompson

### Overview
This repo contains all the input data, code, and output required for our research. If you wish to execute code, the python scripts are likely the easiest to implement; Colab notebooks require a copy of the original data in your Google Drive.


### Contents
* `code` all code required to perform research. 
  * The script `analysis.py` and notebook `analysis.ipynb` both serve as the primary point of interaction and execution for training models, the latter for Colab. The notebook `transfer.ipynb` similarly manages working with pre-trained models.
  * `data_mgmt.py` manages data on disk. The first time you train a model, this script will organize the appropriate test/train data organization. In the case of 3D data, it writes 100Gb of numpy arrays.
  * `models.py` and `models3D.py` contain functions that when called create, train, and save models. They also contain a wrapper class definition for convenience.
* `data/original` untouched DFDC data. Functions in `data_mgmt.py` will write to the `data` directory when organizing your local environment.
* `report` written report, slides, and recording of presentation.
* `saved_models` all the final models, saved in tensorflow format
* `small_local` point functions here for small local testing
* `visualization` 
  * `visuals.ipynb` script for making training diagram
  * `model_diagrams` images of different network architectures
  * `plotting_data` all the train/test logs used for graphics

