# PyTorch Mask-RCNN for Instance Segmentation

Using Mask-RCNN from Pytorch for instance segmentation with ease.

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Inference](#inference)
- [See Training Results](#see-training-results)


-> This project works (for example) perfectly with the [3xM Datasets](https://github.com/xXAI-botXx/3xM)



---
### Installation

1. Install Anaconda
2. [Clone or Download this project](https://github.com/xXAI-botXx/torch-mask-rcnn-instance-segmentation)<br>
    ```terminal
    git clone https://github.com/xXAI-botXx/torch-mask-rcnn-instance-segmentation.git
    ```
3. Install an environment for instance segmentation:<br>
    ```terminal
    cd /to/git/project
    conda env create -f conda_env.yml
    ```

For Windows there are 2 other yml files, you can try if one of them works for you.

In worse case you also can setup your own conda env:
```terminal
conda env create python=3.10 -n maskrcnn -y
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install pillow -y
conda install opencv -c conda-forge -y
conda install matplotlib -y
conda install scipy -y
conda install mlflow -y
conda install tensorboard -y
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
EXTENDED_VERSION = True
    WEIGHTS_PATH = None         # Path to the model weights file
    USE_DEPTH = True            # Whether to include depth information -> as rgb and depth on green channel
    VERIFY_DATA = True         # True is recommended

    GROUND_PATH = "/mnt/morespace/3xM"     
    DATASET_NAME = "3xM_Dataset_160_160"
    IMG_DIR = os.path.join(GROUND_PATH, DATASET_NAME, 'rgb')        # Directory for RGB images
    DEPTH_DIR = os.path.join(GROUND_PATH, DATASET_NAME, 'depth')    # Directory for depth-preprocessed images
    MASK_DIR = os.path.join(GROUND_PATH, DATASET_NAME, 'mask')      # Directory for mask-preprocessed images

    DATA_MODE = DATA_LOADING_MODE.ALL  # Mode for loading data -> All, Random, Range, Single Image
    AMOUNT = 100                       # Number of images for random mode
    START_IDX = 0                      # Starting index for range mode
    END_IDX = 99                       # Ending index for range mode
    IMAGE_NAME = "3xM_0_10_10.png"     # Specific image name for single mode

    NUM_WORKERS = 4                    # Number of workers for data loading

    MULTIPLE_DATASETS = None                                        # Path to folder for training multiple models
    SKIP_DATASETS = ["3xM_Test_Datasets", "3xM_Dataset_10_160"]     # Datasets to skip, if training multiple datasets
    NAME = 'extended_mask_rcnn_rgbd'                                # Name of the model to use

    USING_EXPERIMENT_TRACKING = False              # Enable experiment tracking
    CREATE_NEW_EXPERIMENT = True                   # Whether to create a new experiment run
    EXPERIMENT_NAME = "3xM Instance Segmentation"  # Name of the experiment

    NUM_EPOCHS = 100                   # Number of training epochs
    LEARNING_RATE = 3e-3               # Learning rate for the optimizer
    MOMENTUM = 0.9                     # Momentum for the optimizer
    BATCH_SIZE = 1                     # Batch size for training
    SHUFFLE = True                     # Shuffle the data during training
    
    # Decide which Data Augmentation should be applied
    APPLY_RANDOM_FLIP = True
    APPLY_RANDOM_ROTATION = True
    APPLY_RANDOM_CROP = True
    APPLY_RANDOM_BRIGHTNESS_CONTRAST = True
    APPLY_RANDOM_GAUSSIAN_NOISE = True
    APPLY_RANDOM_GAUSSIAN_BLUR = True
    APPLY_RANDOM_SCALE = True
    APPLY_RANDOM_BACKGROUND_MODIFICATION = True
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
EXTENDED_VERSION = False      # Decides if a Mask RCNN with more FPN Layers and a less strict NMS and another Learning Rate strategy should get used
WEIGHTS_PATH = "./weights/mask_rcnn_rgb_3xM_Dataset_80_160_epoch_040.pth"  # Path to the model weights file
MASK_SCORE_THRESHOLD = 0.5    # score to accept a result mask
USE_DEPTH = False             # Whether to include depth information -> as rgb and depth on green channel
VERIFY_DATA = True            # True is recommended

GROUND_PATH = "D:/3xM/3xM_Test_Dataset/"
DATASET_NAME = "OCID-dataset-prep"
IMG_DIR = os.path.join(GROUND_PATH, DATASET_NAME, 'rgb')        # Directory for RGB images
DEPTH_DIR = os.path.join(GROUND_PATH, DATASET_NAME, 'depth')    # Directory for depth-preprocessed images
MASK_DIR = os.path.join(GROUND_PATH, DATASET_NAME, 'mask')      # Directory for mask-preprocessed images

DATA_MODE = DATA_LOADING_MODE.ALL  # Mode for loading data -> All, Random, Range, Single Image
AMOUNT = 10                        # Number of images for random mode
START_IDX = 0                      # Starting index for range mode
END_IDX = 10                       # Ending index for range mode
IMAGE_NAME = "3xM_10000_10_80.png" # Specific image name for single mode

NUM_WORKERS = 4                    # Number of workers for data loading

OUTPUT_DIR = "./output"            # Directory to save output files
USE_MASK = True                    # Whether to use masks during inference
SHOULD_SAVE_MASK = False           # Decides whether to save the result mask
OUTPUT_TYPE = "png"                # Output format: 'numpy-array' or 'png'
SHOULD_VISUALIZE_MASK = False,     # if the predicted mask should get visualized
SHOULD_VISUALIZE_MASK_AND_IMAGE = False,    # if the mask on top of the image get visualized
SAVE_VISUALIZATION = False          # Save the visualizations to disk
SHOW_VISUALIZATION = False          # Display the visualizations
SAVE_EVALUATION = True              # Save the evaluation results
SHOW_EVALUATION = False             # Display the evaluation results
SHOW_INSIGHTS = False               # If the insight pictures should get showed, from the in-between network steps
SAVE_INSIGHTS = False               # If the insight pictures should get saved, from the in-between network steps

RESET_OUTPUT = True                 # Should output folder be deleted/cleared before inferencing 
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






