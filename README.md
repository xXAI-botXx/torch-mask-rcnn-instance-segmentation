# PyTorch Mask-RCNN for Instance Segmentation

Using Mask-RCNN from Pytorch for instance segmentation with ease.

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Inference](#inference)
- [See Training Results](#see-training-results)



---
### Installation

1. Install Anaconda
2. Install an environment for instance segmentation:<br>
    ```terminal
    conda env create -f conda_env.yml
    ```
3. [Download this project](https://github.com/xXAI-botXx/torch-mask-rcnn-instance-segmentation)<br>
    ```terminal
    git clone https://github.com/xXAI-botXx/torch-mask-rcnn-instance-segmentation.git
    ```



---
### Data Preparation

You will need 2 - 3 folders:
rgb, mask, depth (if wanted)

And every image, mask and depth image which are the same should have the same name.

```plaintext
|--rgb
|--------my_data_0.png
|--------my_data_1.png
|--------...
|--mask
|--------my_data_0.png
|--------my_data_1.png
|--------...
|--depth
|--------my_data_0.png
|--------my_data_1.png
|--------...
```

The mask can be provided as rgb image, where every unique value stands for an unique object. The background will be auto detected. This will lead to poor performance. Instead it is recommended to provide the masks as gray image, where (again) every unique value belongs to one object in the image.

The depth image can be provided as grey image or as rgb image. If you are give the depth image as rgb image, make sure that the image is encoded in the green channel!

In general you should provide your images as png files. This project is not tested for other image types and can propably not work (because of the alpha channel from png files). PNG is also without information loss, so it is a recommended format.



---
### Training

To train your own model you have to open the **maskrcnn_toolkit.py** and go to the

```plaintext
#############
# variables #
#############
```

section.
There you change the variable MODE to RUN_MODE.TRAIN.

Then scroll a bit down to

```plaintext
# -------- #
# TRAINING #
# -------- #
```

Now you can set your own settings. Following settings are available:

```python
WEIGHTS_PATH = None                             # Path to the model weights file
USE_DEPTH = False                               # Whether to include depth information -> as rgb and depth on green channel

IMG_DIR ='.../3xM_Dataset_10_10/rgb'            # Directory for RGB images
DEPTH_DIR = '.../3xM_Dataset_10_10/depth'       # Directory for depth-preprocessed images
MASK_DIR = '.../3xM_Dataset_10_10/mask-prep'    # Directory for mask-preprocessed images
WIDTH = 1920                                    # Image width for processing
HEIGHT = 1080                                   # Image height for processing

DATA_MODE = DATA_LOADING_MODE.ALL               # Mode for loading data -> All, Random, Range, Single Image
AMOUNT = 100                                    # Number of images for random mode
START_IDX = 0                                   # Starting index for range mode
END_IDX = 99                                    # Ending index for range mode
IMAGE_NAME = "3xM_0_10_10.png"                  # Specific image name for single mode

NUM_WORKERS = 4                                 # Number of workers for data loading

MULTIPLE_DATASETS = None                        # Path to folder for training multiple models
NAME = 'mask_rcnn_TEST'                         # Name of the model to use

USING_EXPERIMENT_TRACKING = True                # Enable experiment tracking
CREATE_NEW_EXPERIMENT = False                   # Whether to create a new experiment run
EXPERIMENT_NAME = "3xM Instance Segmentation"   # Name of the experiment

NUM_EPOCHS = 20                                 # Number of training epochs
LEARNING_RATE = 0.001                           # Learning rate for the optimizer
MOMENTUM = 0.9                                  # Momentum for the optimizer
DECAY = 0.0005                                  # Weight decay for regularization
BATCH_SIZE = 8                                  # Batch size for training
SHUFFLE = True                                  # Shuffle the data during training

# Decide which Data Augmentation should be applied
APPLY_RANDOM_FLIP = True
APPLY_RANDOM_ROTATION = True
APPLY_RANDOM_CROP = True
APPLY_RANDOM_BRIGHTNESS_CONTRAST = True
APPLY_RANDOM_GAUSSIAN_NOISE = True
APPLY_RANDOM_GAUSSIAN_BLUR = True
APPLY_RANDOM_SCALE = True
```


You also can run a hyperparameter optimization.
For that you have to set MODE to RUN_MODE.HYPERPARAMETER_OPTIMIZATION and go to the

```plaintext
# --------------------- #
# HYPERPARAMETER TUNING #
# --------------------- #
```

section.

There are following parameters:

```python
USE_DEPTH = False                               # Whether to include depth information -> as rgb and depth on green channel

IMG_DIR ='.../3xM_Dataset_10_10/rgb'            # Directory for RGB images
DEPTH_DIR = '.../3xM_Dataset_10_10/depth-prep'  # Directory for depth-preprocessed images
MASK_DIR = '.../3xM_Dataset_10_10/mask'         # Directory for mask-preprocessed images
WIDTH = 800                                     # Image width for processing
HEIGHT = 450                                    # Image height for processing

DATA_MODE = DATA_LOADING_MODE.RANGE             # Mode for loading data -> All, Random, Range, Single Image
AMOUNT = 100                                    # Number of images for random mode
START_IDX = 0                                   # Starting index for range mode
END_IDX = 199                                   # Ending index for range mode
IMAGE_NAME = "3xM_0_10_10.png"                  # Specific image name for single mode

NUM_WORKERS = 4                                 # Number of workers for data loading

# Decide which Data Augmentation should be applied
APPLY_RANDOM_FLIP = True
APPLY_RANDOM_ROTATION = True
APPLY_RANDOM_CROP = True
APPLY_RANDOM_BRIGHTNESS_CONTRAST = True
APPLY_RANDOM_GAUSSIAN_NOISE = True
APPLY_RANDOM_GAUSSIAN_BLUR = True
APPLY_RANDOM_SCALE = True
```


---
### Inference

For Inference set MODE to RUN_MODE.INFERENCE and go to the

```python
# --------- #
# INFERENCE #
# --------- #
```

section.

Now you can adjust the setting after your needs:

```python
WEIGHTS_PATH = "./weights/mask_rcnn.pth"        # Path to the model weights file
USE_DEPTH = False                               # Whether to include depth information -> as rgb and depth on green channel

IMG_DIR ='.../3xM_Dataset_10_10/rgb-prep'       # Directory for RGB images
DEPTH_DIR = '.../3xM_Dataset_10_10/depth-prep'  # Directory for depth-preprocessed images
MASK_DIR = '.../3xM_Dataset_10_10/mask-prep'    # Directory for mask-preprocessed images
WIDTH = 800                                     # Image width for processing
HEIGHT = 450                                    # Image height for processing

DATA_MODE = DATA_LOADING_MODE.SINGLE            # Mode for loading data -> All, Random, Range, Single Image
AMOUNT = 10                                     # Number of images for random mode
START_IDX = 0                                   # Starting index for range mode
END_IDX = 9                                     # Ending index for range mode
IMAGE_NAME = "3xM_0_10_10.png"                  # Specific image name for single mode

NUM_WORKERS = 4                                 # Number of workers for data loading

OUTPUT_DIR = "./output"                         # Directory to save output files
USE_MASK = True                                 # Whether to use masks during inference
OUTPUT_TYPE = "png"                             # Output format: 'numpy-array' or 'png'
SHOULD_VISUALIZE = True                         # Whether to visualize the results
VISUALIZATION_DIR = "./output/visualizations"   # Directory to save visualizations
SAVE_VISUALIZATION = True                       # Save the visualizations to disk
SHOW_VISUALIZATION = False                      # Display the visualizations
SAVE_EVALUATION = True                          # Save the evaluation results
SHOW_EVALUATION = False                         # Display the evaluation results
```

Alternatively you can use a basic and very simple inference. Here you will give an image and the inference will only make the inference + improving the results and will return the results.
Set MODE to SIMPLE_INFERENCE And go to the

```python
# ---------------- #
# SIMPLE INFERENCE #
# ---------------- #
```

section.

With following parameters:

```python
WEIGHTS_PATH = "./weights/maskrcnn.pth"         # Path to the model weights file
USE_DEPTH = False                               # Whether to include depth information -> as rgb and depth on green channel

IMG_DIR ='.../3xM_Dataset_1_1_TEST/'            # Directory for RGB images
DEPTH_DIR = '.../3xM_Dataset_1_1_TEST/depth'    # Directory for depth-preprocessed images
WIDTH = 1920                                    # Image width for processing
HEIGHT = 1080                                   # Image height for processing

IMAGE_NAME = "3xM_0_10_10.png"                  # Specific image name 

OUTPUT_DIR = "./output"                         # Directory to save output files
USE_MASK = True                                 # Whether to use masks during inference
OUTPUT_TYPE = "png"                             # Output format: 'numpy-array' or 'png'
SHOULD_SAVE = True                              # Decides whether to save the mask or not -> mask will be returned
```


---
### See Training Results

If you used experiment tracking, you can see the results as following described:


**Start Tensorboard**

1. ``` terminal
   conda activate maskrcnn
   ```

2. ``` terminal
   tensorboard --logdir=~/src/instance-segmentation/runs --host=0.0.0.0
   ```

    Change the path to your project.

3. Now open the browser and type:<br>
   http://10.24.3.16:6006 (if you are using a remote connection)<br>
   http://126.0.0.1:6006 (if you are running it locally)



**Start MLFlow UI**

1. ``` terminal
   conda activate maskrcnn
   ```

2. ``` terminal
   mlflow ui --host=0.0.0.0 --backend-store-uri file:///home/local-admin/src/instance-segmentation/mlruns
   ```

3. Now open the browser and type:<br>
   http://10.24.3.16:5000 (if you are using a remote connection)<br>
   http://126.0.0.1:5000 (if you are running it locally)



Hint: You can search for your mlflow directory with following command (navigate to your home folder before):

```bash
find . -type d -name mlruns 2>/dev/null
```






