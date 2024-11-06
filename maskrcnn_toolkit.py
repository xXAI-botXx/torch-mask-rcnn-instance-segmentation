"""
Instance Segmentation Toolkit with Mask-RCNN

Tags:
- training
- inference

- data augmentation
- dual dir data loading (over ID, same name)
- control the program by changing the variables below
- experiment tracking
- tensorboard
- logging
- visualizing

- depth image integration (3 or 4 channels)

- functionalities for training and inference



This script defines a set of variables to configure either a Mask-RCNN model training or inference task for instance segmentation.
Depending on the value of `SHOULD_TRAIN`, the script will either train the model on the specified dataset(s) or run inference on 
a set of images.


Usage:
------
Scroll a little bit down to the **variables** section and put the values to your need.
The first 2 Variables decides your mode and 

1. To train the model:
    Set `MODE = RUN_MODE.TRAIN` and provide paths to your data, mask, and (optionally) depth images.
    The script will train the Mask-RCNN model according to the specified parameters.

2. To run inference:
    Set `MODE = RUN_MODE.INFERENCE` and provide the path to the weights and images for testing.
    The script will run inference and optionally visualize/save the results.

    You can choose between a simple inference and a more advanced inference. 
    The simple inference just inference one given image and save + returns the mask.
    The other inference can inference multiple images (also one) and can visualize the results and evaluate them.



By Tobia Ippolito <3
"""

###############
# definitions #
###############
import os
from enum import Enum

class DATA_LOADING_MODE(Enum):
    ALL = "all"
    RANGE = "range"
    RANDOM = "random"
    SINGLE = "single"



class RUN_MODE(Enum):
    TRAIN = "train"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    INFERENCE = "inference"






#############
# variables #
#############
# Change these variables to your need

MODE = RUN_MODE.TRAIN



# -------- #
# TRAINING #
# -------- #
if MODE == RUN_MODE.TRAIN:
    WEIGHTS_PATH = None  # Path to the model weights file
    USE_DEPTH = True                   # Whether to include depth information -> as rgb and depth on green channel
    VERIFY_DATA = False         # True is recommended

    GROUND_PATH = "/mnt/morespace/3xM"    # "/mnt/morespace/3xM" "D:/3xM" 
    DATASET_NAME = "3xM_Dataset_80_80"
    IMG_DIR = os.path.join(GROUND_PATH, DATASET_NAME, 'rgb')        # Directory for RGB images
    DEPTH_DIR = os.path.join(GROUND_PATH, DATASET_NAME, 'depth')    # Directory for depth-preprocessed images
    MASK_DIR = os.path.join(GROUND_PATH, DATASET_NAME, 'mask')      # Directory for mask-preprocessed images
    WIDTH = 1920   # 1920, 1024, 800, 640                # Image width for processing
    HEIGHT = 1080  # 1080, 576, 450, 360                    # Image height for processing

    DATA_MODE = DATA_LOADING_MODE.ALL  # Mode for loading data -> All, Random, Range, Single Image
    AMOUNT = 100                       # Number of images for random mode
    START_IDX = 0                      # Starting index for range mode
    END_IDX = 99                       # Ending index for range mode
    IMAGE_NAME = "3xM_0_10_10.png"     # Specific image name for single mode

    NUM_WORKERS = 4                    # Number of workers for data loading

    MULTIPLE_DATASETS = None    # "/mnt/morespace/3xM"           # Path to folder for training multiple models
    SKIP_DATASETS = ["3xM_Test_Datasets"]
    NAME = 'mask_rcnn_rgbd_TEST_cycle'                 # Name of the model to use

    USING_EXPERIMENT_TRACKING = True   # Enable experiment tracking
    CREATE_NEW_EXPERIMENT = True       # Whether to create a new experiment run
    EXPERIMENT_NAME = "3xM Instance Segmentation"  # Name of the experiment

    NUM_EPOCHS = 100                    # Number of training epochs
    LEARNING_RATE = 1e-7              # Learning rate for the optimizer
    # MOMENTUM = 0.9                     # Momentum for the optimizer
    DECAY = 0.0005                     # Weight decay for regularization
    BATCH_SIZE = 5                    # Batch size for training
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
    
    # MASK_SCORE_THRESHOLD = 0.9



# --------------------- #
# HYPERPARAMETER TUNING #
# --------------------- #
if MODE == RUN_MODE.HYPERPARAMETER_TUNING:
    USE_DEPTH = False                   # Whether to include depth information -> as rgb and depth on green channel
    VERIFY_DATA = False         # True is recommended

    GROUND_PATH = "D:/3xM"    # "/mnt/morespace/3xM"
    DATASET_NAME = "3xM_Dataset_160_80"
    IMG_DIR = os.path.join(GROUND_PATH, DATASET_NAME, 'rgb')        # Directory for RGB images
    DEPTH_DIR = os.path.join(GROUND_PATH, DATASET_NAME, 'depth')    # Directory for depth-preprocessed images
    MASK_DIR = os.path.join(GROUND_PATH, DATASET_NAME, 'mask')      # Directory for mask-preprocessed images
    WIDTH = 1920   # 1920, 1024, 800, 640                # Image width for processing
    HEIGHT = 1080  # 1080, 576, 450, 360                    # Image height for processing

    DATA_MODE = DATA_LOADING_MODE.ALL  # Mode for loading data -> All, Random, Range, Single Image
    AMOUNT = 100                       # Number of images for random mode
    START_IDX = 0                      # Starting index for range mode
    END_IDX = 499                       # Ending index for range mode
    IMAGE_NAME = "3xM_0_10_10.png"     # Specific image name for single mode

    NUM_WORKERS = 4                    # Number of workers for data loading
    BATCH_SIZE = 5
    # MOMENTUM = 0.9                     # Momentum for the optimizer
    DECAY = 0.0005                     # Weight decay for regularization
    
    # Decide which Data Augmentation should be applied
    APPLY_RANDOM_FLIP = True
    APPLY_RANDOM_ROTATION = True
    APPLY_RANDOM_CROP = True
    APPLY_RANDOM_BRIGHTNESS_CONTRAST = True
    APPLY_RANDOM_GAUSSIAN_NOISE = True
    APPLY_RANDOM_GAUSSIAN_BLUR = True
    APPLY_RANDOM_SCALE = True
    APPLY_RANDOM_BACKGROUND_MODIFICATION = True
    
    # MASK_SCORE_THRESHOLD = 0.9
    


# --------- #
# INFERENCE #
# --------- #
if MODE == RUN_MODE.INFERENCE:
    WEIGHTS_PATH = "./weights/mask_rcnn_rgbd_3xM_Dataset_80_80.pth"  # Path to the model weights file
    MASK_SCORE_THRESHOLD = 0.9
    USE_DEPTH = True                   # Whether to include depth information -> as rgb and depth on green channel
    VERIFY_DATA = False         # True is recommended

    GROUND_PATH = "D:/3xM"    # "/mnt/morespace/3xM"
    DATASET_NAME = "3xM_Dataset_160_80"
    IMG_DIR = os.path.join(GROUND_PATH, DATASET_NAME, 'rgb')        # Directory for RGB images
    DEPTH_DIR = os.path.join(GROUND_PATH, DATASET_NAME, 'depth')    # Directory for depth-preprocessed images
    MASK_DIR = os.path.join(GROUND_PATH, DATASET_NAME, 'mask')      # Directory for mask-preprocessed images
    WIDTH = 800                       # Image width for processing
    HEIGHT = 450                      # Image height for processing

    DATA_MODE = DATA_LOADING_MODE.SINGLE  # Mode for loading data -> All, Random, Range, Single Image
    AMOUNT = 10                       # Number of images for random mode
    START_IDX = 0                      # Starting index for range mode
    END_IDX = 9                       # Ending index for range mode
    IMAGE_NAME = "3xM_0_10_10.png"     # Specific image name for single mode

    NUM_WORKERS = 4                    # Number of workers for data loading

    OUTPUT_DIR = "./output"            # Directory to save output files
    USE_MASK = True                    # Whether to use masks during inference
    OUTPUT_TYPE = "png"                # Output format: 'numpy-array' or 'png'
    SHOULD_VISUALIZE = True            # Whether to visualize the results
    VISUALIZATION_DIR = "./output/visualizations"  # Directory to save visualizations
    SAVE_VISUALIZATION = True          # Save the visualizations to disk
    SHOW_VISUALIZATION = False          # Display the visualizations
    SAVE_EVALUATION = True             # Save the evaluation results
    SHOW_EVALUATION = False             # Display the evaluation results

    SHOW_INSIGHTS = False
    SAVE_INSIGHTS = True






###########
# imports #
###########

# basics
from datetime import datetime, timedelta
import time
from IPython.display import clear_output
from functools import partial
import statistics

# image
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2 
from PIL import Image    # for PyTorch Transformations

# deep learning
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam    #, SGD
from torch.optim.lr_scheduler import OneCycleLR, CyclicLR

import torchvision
from torchvision.models.detection import MaskRCNN
# from torchvision.models.detection import maskrcnn_resnet50_fpn
# from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models import ResNet50_Weights
# from torchvision.transforms import functional as F
import torchvision.transforms as T

# experiment tracking
from torch.utils.tensorboard import SummaryWriter
import mlflow
import mlflow.pytorch

# optimization
import optuna
from scipy.optimize import linear_sum_assignment


###########
# general #
###########

def load_maskrcnn(weights_path=None, use_4_channels=False, pretrained=True,
                  image_mean=[0.485, 0.456, 0.406, 0.5], image_std=[0.229, 0.224, 0.225, 0.5],    # from ImageNet
                  min_size=1080, max_size=1920, log_path=None, should_log=False, should_print=True):
    """
    Load a Mask R-CNN model with a specified backbone and optional modifications.

    This function initializes a Mask R-CNN model with a ResNet50-FPN backbone. 
    It allows for the configuration of input channels and loading of pretrained weights.

    Parameters:
    -----------
        weights_path (str, optional): 
            Path to the weights file to load the model's state dict. 
            If None, the model will be initialized with random weights or 
            pretrained weights if 'pretrained' is True.
        
        use_4_channels (bool, optional): 
            If True, modifies the first convolutional layer to accept 
            4 input channels instead of the default 3. The weights from 
            the existing channels are copied accordingly.
        
        pretrained (bool, optional): 
            If True, loads the pretrained weights for the backbone. 
            Defaults to True.

    Returns:
    --------
        model (MaskRCNN): 
            The initialized Mask R-CNN model instance, ready for training or inference.
    """
    backbone = resnet_fpn_backbone(backbone_name='resnet50', weights=ResNet50_Weights.IMAGENET1K_V2) # ResNet50_Weights.IMAGENET1K_V1)
    model = MaskRCNN(backbone, num_classes=2)  # 2 Classes (Background + 1 Object)
    # odel = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT, num_classes=2)    # ResNet50_Weights.DEFAULT

    if use_4_channels:
        # Change the first Conv2d-Layer for 4 Channels
        in_features = model.backbone.body.conv1.in_channels    # this have to be changed
        out_features = model.backbone.body.conv1.out_channels
        kernel_size = model.backbone.body.conv1.kernel_size
        stride = model.backbone.body.conv1.stride
        padding = model.backbone.body.conv1.padding
        
        # Create new conv layer with 4 channels
        new_conv1 = torch.nn.Conv2d(4, out_features, kernel_size=kernel_size, stride=stride, padding=padding)
        
        # copy the existing weights from the first 3 Channels
        with torch.no_grad():
            new_conv1.weight[:, :3, :, :] = model.backbone.body.conv1.weight  # Copy old 3 Channels
            new_conv1.weight[:, 3:, :, :] = model.backbone.body.conv1.weight[:, :1, :, :]  # Init new 4.th Channel with the one old channel
            # torch.nn.init.kaiming_normal_(new_conv1.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')

        
        model.backbone.body.conv1 = new_conv1  # Replace the old Conv1 Layer with the new one
        
        # Modify the transform to handle 4 channels
        #       - Replace the transform in the model with a custom one
        model.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
        
    if weights_path:
        model.load_state_dict(state_dict=torch.load(weights_path, weights_only=True)) 
    
    model_str = "Parameter of Mask R-CNN:"
    model_parts = dict()
    for name, param in model.named_parameters():
        model_str += f"\n    - Parameter Name: {name}"
        model_str += f"\n    - Parameter Shape: {param.shape}"
        model_str += f"\n    - Requires Grad: {param.requires_grad}"
        
        if "backbone" in name:
            model_str += f"\n    - Belongs to the Backbone\n"
        elif "rpn" in name:
            model_str += f"\n    - Belongs to the RPN\n"
        elif "roi_heads" in name:
            model_str += f"\n    - Belongs to the ROI Heads\n"
        else:
            model_str += f"\n    - Belongs to another part of the model\n"
            
        model_str += f"\n              {'-'*50}"
        
        # get model part
        model_part_name_1 = name.split(".")[0]
        model_part_name_2 = name.split(".")[1]
        model_part_name = f"{model_part_name_1.upper()} - {model_part_name_2.upper()}"
        if model_part_name in model_parts.keys():
            model_parts[model_part_name] += 1
        else:
            model_parts[model_part_name] = 1
            
    model_str += "\n\nParameter Summary:"
    for key, value in model_parts.items():
        distance = 40 - len(f"    - {key}")
        model_str += f"\n    - {key}{' '*distance}({value} parameters)"
    model_str += f"\n{'-'*64}\n"
    
        
    log(log_path, model_str, should_log=should_log, should_print=should_print)
    
    return model



def clear_printing():
    # terminal
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # notebook
    clear_output()



class Dual_Dir_Dataset(Dataset):
    """
    A dataset class for loading and processing images, depth maps, and masks 
    from specified directories. This dataset supports various modes of data loading 
    including random selection, range-based selection, and single image loading.


    Parameters:
    -----------
    img_dir (str): Directory containing RGB images.
    depth_dir (str, optional): Directory containing depth images. Default is None.
    mask_dir (str, optional): Directory containing mask images. Default is None.
    transform (callable, optional): Optional transform to be applied to the images. Default is None.
    amount (int): Number of random images to select when using random mode. Default is 100.
    start_idx (int): Starting index for range mode. Default is 0.
    end_idx (int): Ending index for range mode. Default is 99.
    image_name (str): The name of a single image for single mode. Default is "3xM_0_10_10.jpg".
    data_mode (str): Mode for loading images (e.g., 'ALL', 'RANDOM', 'RANGE', 'SINGLE'). Default is DATA_LOADING_MODE.ALL.
    use_mask (bool): Flag indicating whether to load mask images. Default is True.
    use_depth (bool): Flag indicating whether to load depth images. Default is False.
    log_path (str, optional): Path for logging data verification results. Default is None.
    width (int): Width to which images should be resized (if resizing is implemented). Default is 1920.
    height (int): Height to which images should be resized (if resizing is implemented). Default is 1080.


    Attributes:
    -----------
    img_names (list): List of image names based on the loading mode and parameters.
    

    Methods:
    --------
    __len__(): Returns the total number of images in the dataset.
    __getitem__(idx): Loads and returns the image, depth map, and mask at the specified index.
    get_bounding_boxes(masks): Computes and returns bounding boxes from the mask images.
    verify_data(): Verifies the existence and validity of images, masks, and depth maps.
    load_datanames(path_to_images, amount, start_idx, end_idx, image_name, data_mode): 
        Loads file paths from a specified directory based on the given mode.
    verify_mask_post_processing(original_mask, new_mask): 
        Validates that the transformation from RGB to grey mask preserves the correct number of objects.
    rgb_mask_to_grey_mask(rgb_img, verify): 
        Converts an RGB mask to a grey mask, mapping unique RGB values to increasing integers.

    Raises:
    -------
    ValueError: If an invalid data_mode is provided or if the verification of mask processing fails.
    """

    def __init__(self, 
                img_dir, 
                depth_dir=None,
                mask_dir=None, 
                transform=None,
                amount=100,     # for random mode
                start_idx=0,    # for range mode
                end_idx=99,     # for range mode
                image_name="3xM_0_10_10.jpg", # for single mode
                data_mode=DATA_LOADING_MODE.ALL,
                use_mask=True,
                use_depth=False,
                log_path=None,
                width=1920,
                height=1080,
                should_print=True,
                should_log=True,
                should_verify=True
                ):

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.depth_dir = depth_dir
        self.transform = transform
        self.use_mask = use_mask
        self.use_depth = use_depth
        self.log_path = log_path
        self.width = width
        self.height = height
        self.should_print = should_print
        self.should_log = should_log,
        self.background_value = 1    # will be auto setted in verification
        self.img_names = self.load_datanames(
                                        path_to_images=img_dir, 
                                        amount=amount, 
                                        start_idx=start_idx, 
                                        end_idx=end_idx, 
                                        image_name=image_name, 
                                        data_mode=data_mode
                                        )
        
        if should_verify:
            self.verify_data()

        # update augmentations -> needed for background augmentation
        self.transform.update(bg_value=self.background_value)



    def __len__(self):
        return len(self.img_names)



    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        depth_path = os.path.join(self.depth_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # Load the RGB Image, Depth Image and the Gray Mask (and make sure that the size is right)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = resize_with_padding(image, target_h=self.height, target_w=self.width, method=cv2.INTER_LINEAR)
        
        if self.use_depth:
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if len(depth.shape) > 2:
                _, depth, _, _ =  cv2.split(depth)
            # depth = resize_with_padding(depth, target_h=self.height, target_w=self.width, method=cv2.INTER_LINEAR)

        if self.use_mask:
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)    # cv2.IMREAD_GRAYSCALE)
            if len(mask.shape) > 2:
                mask = self.rgb_mask_to_grey_mask(rgb_img=mask, verify=False)
            # mask = resize_with_padding(mask, target_h=self.height, target_w=self.width, method=cv2.INTER_NEAREST)

        # Apply transformations
        if self.transform:
            if self.use_mask and self.use_depth:
                image, depth, mask = self.transform(image, depth, mask)
            elif self.use_mask:
                image, _, mask = self.transform(image, None, mask)
            elif self.use_depth:
                image, depth, _ = self.transform(image, depth, None)
            else:
                image, _, _ = self.transform(image, None, None)

        # Add Depth to Image as 4th Channel
        if self.use_depth:
            # Ensure depth is a 2D array
            if depth.ndim == 2:
                depth = np.expand_dims(depth, axis=-1)  # Add an extra dimension for depth channel
            
            # Concatenate along the channel axis to form an RGBA image (4th channel is depth)
            image = np.concatenate((image, depth), axis=-1)
        
        # image to tensor
        image = T.ToTensor()(image)

        if self.use_mask:
           
            # check objects in masks -> if empty than create empty mask
            obj_ids = np.unique(mask)
            obj_ids = obj_ids[obj_ids != self.background_value]
            if len(obj_ids) <= 0:  
                target = {
                    'boxes': torch.zeros((0, 4), dtype=torch.float32),  # No bounding boxes
                    'labels': torch.zeros((0,), dtype=torch.int64),     # No labels
                    'masks': torch.zeros((0, self.height, self.width), dtype=torch.uint8),  # No masks
                }
            else:
                 # Create List of Binary Masks
                masks = np.zeros((len(obj_ids), mask.shape[0], mask.shape[1]), dtype=np.uint8)

                for i, obj_id in enumerate(obj_ids):
                    if obj_id == self.background_value:  # Background
                        continue
                    masks[i] = (mask == obj_id).astype(np.uint8)

                # Convert the masks to a torch tensor
                masks = torch.as_tensor(masks, dtype=torch.uint8)
            
                target = {
                    "masks": masks,
                    "boxes": self.get_bounding_boxes(masks),
                    "labels": torch.ones(masks.shape[0], dtype=torch.int64)  # Set all IDs to 1 -> just one class type
                }
                
            # FIXME for debugging
            # fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(20, 15), sharey=True)
            # fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.02, hspace=None)
            
            # # plot original image
            # print(image.shape)
            # print(target["masks"].shape)
            # ax[0].imshow(np.transpose(image.cpu().numpy(), (1, 2, 0)))
            # ax[0].set_title("Original")
            # ax[0].axis("off")

            # # plot mask alone
            # extract_and_visualize_mask(np.transpose(target["masks"].cpu().numpy(), (1, 2, 0)), image=None, ax=ax[1], visualize=True)
            # ax[1].set_title("Prediction Mask")
            # ax[1].axis("off")
            
            # plt.savefig("./DEBUGGING_PLOT.jpg")
            # plt.clf()
            
            # FIXME debugging end
            
            return image, target, img_name
        else:
            return image, img_name



    def get_bounding_boxes(self, masks):
        boxes = []
        
        for mask in masks:
            # object = np.unique(mask[mask != 0])
            # if len(object) > 1:
            pos = np.where(mask == 1)
            x_min = np.min(pos[1])
            x_max = np.max(pos[1])
            y_min = np.min(pos[0])
            y_max = np.max(pos[0])
            
            # update x values, if bounding box is too small
            if x_min == x_max:
                warning_str = "WARNING! Too small X in Bounding Box found!"
                if x_max >= self.width:
                    x_min -= 1
                    warning_str += f" X_Min got udated to {x_min} from {x_min+1}"
                else:
                    x_max += 1
                    warning_str += f" X_Max got udated to {x_max} from {x_max-1}"
                log(self.log_path, warning_str, should_log=self.should_log, should_print=self.should_print)
                    
            # update y values, if bounding box is too small
            if y_min == y_max:
                warning_str = "WARNING! Too small Y in Bounding Box found!"
                if y_max >= self.height:
                    y_min -= 1
                    warning_str += f" Y_Min got udated to {y_min} from {y_min+1}"
                else:
                    y_max += 1
                    warning_str += f" Y_Max got udated to {y_max} from {y_max-1}"
                log(self.log_path, warning_str, should_log=self.should_log, should_print=self.should_print)
            
            boxes.append([x_min, y_min, x_max, y_max])
        return torch.as_tensor(boxes, dtype=torch.float32)
    


    def verify_data(self):
        updated_images = []
        log(self.log_path, f"\n{'-'*32}\nVerifying Data...", should_log=self.should_log, should_print=self.should_print)

        images_found = 0
        images_not_found = []

        if self.use_mask:
            masks_found = 0
            masks_not_found = []

        if self.use_depth:
            depth_found = 0
            depth_not_found = []
            
        all_images_size = len(self.img_names)
        
        # for auto background detection
        possible_bg_values = dict()

        for idx, cur_image in enumerate(self.img_names):

            # Check RGB Image
            image_path = os.path.join(self.img_dir, cur_image)
            image_exists = os.path.exists(image_path) and os.path.isfile(image_path) and any([image_path.endswith(ending) for ending in [".png", ".jpg", ".jpeg"]])
            
            if image_exists:
                # rgb_img = cv2.imread(image_path)
                # rgb_shape = rgb_img.shape[:2]  # Height and Width (ignore channels)
                images_found += 1
            else:
                images_not_found += [image_path]

            # Check Depth Image
            if self.use_depth:
                depth_path = os.path.join(self.depth_dir, cur_image)
                depth_exists = os.path.exists(depth_path) and os.path.isfile(depth_path) and any([depth_path.endswith(ending) for ending in [".png", ".jpg", ".jpeg"]])

                if depth_exists:
                    depth_found += 1
                    # if image_exists:
                    #     depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                    #     depth_shape = depth_img.shape[:2]
                    #     if rgb_shape == depth_shape:
                    #         depth_found += 1
                    #     else:
                    #         depth_not_found += [depth_path]
                    # else:
                    #     depth_found += 1
                else:
                    depth_not_found += [depth_path]

            # Check Mask Image
            if self.use_mask:
                mask_path = os.path.join(self.mask_dir, cur_image)
                mask_exists = os.path.exists(mask_path) and os.path.isfile(mask_path) and any([mask_path.endswith(ending) for ending in [".png", ".jpg", ".jpeg"]])
                # check if mask has an object
                # if mask_exists:
                #     mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)    # cv2.IMREAD_GRAYSCALE)
                #     # if len(mask_img.shape) > 2:
                #     #     mask = self.rgb_mask_to_grey_mask(rgb_img=mask_img, verify=False)

                #     if len(np.unique(mask_img)) <= 1:
                #         mask_exists = False
                    # else:
                    #     if image_exists:
                    #         mask_shape = mask_img.shape[:2]

                    #         if rgb_shape != mask_shape:
                    #             mask_exists = False
                
                # get cur bg value (it also could be a large object)
                if mask_exists:
                    mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) 
                    if len(mask_img.shape) > 2:
                        mask = self.rgb_mask_to_grey_mask(rgb_img=mask_img, verify=False)
                        
                    cur_bg = np.min(np.unique(mask_img))
                    if cur_bg in possible_bg_values.keys():
                        possible_bg_values[cur_bg] += 1
                    else:
                        possible_bg_values[cur_bg] = 1
                
                
                if mask_exists:
                    masks_found += 1
                else:
                    masks_not_found += [mask_path]

            # Add Image, if valid
            if self.use_mask and self.use_depth:
                if image_exists and mask_exists and depth_exists:
                    updated_images += [cur_image]
            elif self.use_mask:
                if image_exists and mask_exists:
                    updated_images += [cur_image]
            elif self.use_depth:
                if image_exists and depth_exists:
                    updated_images += [cur_image]
            else:
                if image_exists:
                    updated_images += [cur_image]
                    
            # print
            # progress = round((idx / all_images_size), 2)
            # progress = (idx / all_images_size)
            # if (progress % 0.10) <= 0.001 == 0:
            #     progress_bar_size = 10
            #     progress_bar = int(progress * progress_bar_size)
            #     white_spaces = int(progress_bar_size - progress_bar)
            #     log(self.log_path, f"[{'#'*progress_bar}{' '*white_spaces}]", should_log=self.should_log, should_print=self.should_print)

        # Log/Print Verification Results
        log(self.log_path, f"\n> > > Images < < <\nFound: {round((images_found/len(self.img_names))*100, 2)}% ({images_found}/{len(self.img_names)})", should_log=self.should_log, should_print=self.should_print)
    
        if len(images_not_found) > 0:
            log(self.log_path, "\n Not Found:", should_log=self.should_log, should_print=self.should_print)
            
        for not_found in images_not_found:
            log(self.log_path, f"    -> {not_found}", should_log=self.should_log, should_print=self.should_print)


        if self.use_depth:
            log(self.log_path, f"\n> > > Depth-images < < <\nFound: {round((depth_found/len(self.img_names))*100, 2)}% ({depth_found}/{len(self.img_names)})", should_log=self.should_log, should_print=self.should_print)
                
            if len(depth_not_found) > 0:
                log(self.log_path, "\n Not Found:", should_log=self.should_log, should_print=self.should_print)
                
            for not_found in depth_not_found:
                log(self.log_path, f"    -> {not_found}", should_log=self.should_log, should_print=self.should_print)


        if self.use_mask:
            log(self.log_path, f"\n> > > Masks < < <\nFound: {round((masks_found/len(self.img_names))*100, 2)}% ({masks_found}/{len(self.img_names)})", should_log=self.should_log, should_print=self.should_print)
                
            if len(masks_not_found) > 0:
                log(self.log_path, "\n Not Found:", should_log=self.should_log, should_print=self.should_print)
                
            for not_found in masks_not_found:
                log(self.log_path, f"    -> {not_found}", should_log=self.should_log, should_print=self.should_print)


        log(self.log_path, f"\nUpdating Images...", should_log=self.should_log, should_print=self.should_print)
        log(self.log_path, f"From {len(self.img_names)} to {len(updated_images)} Images\n    -> Image amount reduced by {round(( 1-(len(updated_images)/len(self.img_names)) )*100, 2)}%", should_log=self.should_log, should_print=self.should_print)
        
        # set new updated image ID's
        self.img_names = updated_images
        log(self.log_path, f"{'-'*32}\n", should_log=self.should_log, should_print=self.should_print)

        # detect background
        most_used_bg_value = 1
        most_used_bg_amount = 0 
        if len(possible_bg_values.items()) > 0:
            
            for key, value in possible_bg_values.items():
                if value > most_used_bg_amount:
                    most_used_bg_value = key
                    most_used_bg_amount = value
        self.background_value = most_used_bg_value
        log(self.log_path, f"Auto background Detection: Found '{most_used_bg_value}' as Background value (used in {round((most_used_bg_amount / all_images_size)*100, 2)}% of images as lowest value)", should_log=self.should_log, should_print=self.should_print)



    def load_datanames(
                self,
                path_to_images,
                amount,     # for random mode
                start_idx,  # for range mode
                end_idx,    # for range mode
                image_name, # for single mode
                data_mode=DATA_LOADING_MODE.ALL
            ):
        all_images = os.listdir(path_to_images)

        images = []

        if data_mode.value == DATA_LOADING_MODE.ALL.value:
            data_indices = np.arange(0, len(all_images))
        elif data_mode.value == DATA_LOADING_MODE.RANDOM.value:
            data_indices = np.random.randint(0, len(all_images), amount)
        elif data_mode.value == DATA_LOADING_MODE.RANGE.value:
            end_idx = min(len(all_images)-1, end_idx)
            data_indices = np.arange(start_idx, end_idx+1)
        elif data_mode.value == DATA_LOADING_MODE.SINGLE.value:
            if image_name is None:
                raise ValueError("image_name is None!")
            data_indices = []
            images += [image_name]
        else:
            raise ValueError(f"DATA_MODE has a illegal value: {data_mode}")

        for cur_idx in data_indices:
            images += [all_images[cur_idx]]

        log(self.log_path, f"Image Indices:\n{data_indices}", should_log=self.should_log, should_print=self.should_print)
        log(self.log_path, f"Data Amount: {len(images)}", should_log=self.should_log, should_print=self.should_print)

        # data_amount = len(images)
        return images

    def verify_mask_post_processing(self, original_mask, new_mask):
        # check object amounts
        rgb_mask_unqiue = np.unique(original_mask.reshape(-1, 3), axis=0)
        len_1 = len(rgb_mask_unqiue)

        grey_mask_unqiue = np.unique(new_mask.reshape(-1, 1), axis=0)
        len_2 = len(grey_mask_unqiue)

        if len_1 != len_2:
            raise ValueError(f"Validation failed: The amount of objects are wrong:\n    From {len_1-1} objects to {len_2-1} objects")
        
        # check object pixels
        unique_values_org, counts_org = np.unique(original_mask.reshape(-1, 3), axis=0, return_counts=True)
        unique_values_new, counts_new = np.unique(new_mask.reshape(-1, 1), axis=0, return_counts=True)

        for cur_count_amount in counts_new:
            if not cur_count_amount in counts_org:
                raise ValueError(f"Validation failed: One or more amount of mask-pixel are wrong (the sequence order is not important):\n    -> Original Pixel-amount = {counts_org}\n    -> New Pixel-amount = {counts_new}")
            
        return True



    def rgb_mask_to_grey_mask(self, rgb_img, verify):
        height, width, channels = rgb_img.shape

        # init new mask with only 0 -> everything is background
        grey_mask = np.zeros((height, width), dtype=np.uint8)

        # Get unique RGB values for every row (axis = 0) and before tranform in a simple 2D rgb array
        unique_rgb_values = np.unique(rgb_img.reshape(-1, rgb_img.shape[2]), axis=0)
        
        # Create a mapping from RGB values to increasing integers
        rgb_to_grey = {}
        counter = 1  # Start with 1 since 0 will be reserved for black
        for cur_rgb_value in unique_rgb_values:
            if not np.array_equal(cur_rgb_value, [0, 0, 0]):  # Exclude black
                rgb_to_grey[tuple(cur_rgb_value)] = counter
                counter += 1
            else:
                rgb_to_grey[tuple([0, 0, 0])] = 0

        # Fill the grey mask using the mapping
        for y in range(height):
            for x in range(width):
                rgb_tuple = tuple(rgb_img[y, x])
                grey_mask[y, x] = rgb_to_grey[rgb_tuple] # rgb_to_grey.get(rgb_tuple, 0)  # Default to 0 for black

        # Verify Transaction
        if verify:
            # print("Verify transaction...")
            self.verify_mask_post_processing(original_mask=rgb_img, new_mask=grey_mask)

        # print("Successfull Created a grey mask!")

        return grey_mask



def collate_fn(batch):
    """
    Custom collate function for batching images and targets in a DataLoader.

    This function ensures that each data point in the current batch has the same 
    number of masks by padding the masks with zeros if necessary. It extracts 
    images, targets, and names from the input batch and aligns them to prepare 
    for processing in a neural network.

    Parameters:
    -----------
    batch (list): A list of tuples, where each tuple contains:
        - images (torch.Tensor): The image tensor of shape (C, H, W).
        - targets (dict): A dictionary containing target information for 
          instance segmentation, including:
            - masks (torch.Tensor): A tensor of shape (N, H, W) containing 
              binary masks for each object, where N is the number of objects.
            - boxes (torch.Tensor)
            - classes (torch.Tensor)
        - names (str): The name of the image file.

    Returns:
    --------
    tuple: A tuple containing:
        - images (torch.Tensor): A stacked tensor of shape (B, C, H, W), 
          where B is the batch size.
        - targets (list): A list of dictionaries containing the padded 
          masks, bounding boxes, classes for each image in the batch.
        - names (tuple): A tuple of image names.

    Example:
    --------
    >>> batch = [(image1, target1, name1), (image2, target2, name2)]
    >>> images, targets, names = collate_fn(batch)

    Note:
    -----
    This function assumes that the input batch is non-empty and that each 
    target contains a "masks" key with the corresponding tensor.
    """
    images, targets, names = zip(*batch)
    
    # Find the max number of masks/objects in current batch
    max_num_objs = max(target["masks"].shape[0] for target in targets)
    
    # Add padding
    for target in targets:
        target["masks"] = pad_masks(target["masks"], max_num_objs)
    
    return torch.stack(images, 0), targets, names



# Custom Transformations
class Random_Flip:
    def __call__(self, images):
        rgb_img, depth_img, mask_img = images
        
        if random.random() > 0.6:
            # Horizontal Flipping
            rgb_img = T.functional.hflip(rgb_img)
            if depth_img is not None:
                depth_img = T.functional.hflip(depth_img)
            if mask_img is not None:
                mask_img = T.functional.hflip(mask_img)
        if random.random() > 0.6:
            # Vertical Flipping
            rgb_img = T.functional.vflip(rgb_img)
            if depth_img is not None:
                depth_img = T.functional.vflip(depth_img)
            if mask_img is not None:
                mask_img = T.functional.vflip(mask_img)
        return rgb_img, depth_img, mask_img



class Random_Rotation:
    def __init__(self, max_angle=30):
        self.max_angle = max_angle
    
    def __call__(self, images):
        rgb_img, depth_img, mask_img = images
        
        if random.random() > 0.6:
            angle = random.uniform(-self.max_angle, self.max_angle)
            rgb_img = T.functional.rotate(rgb_img, angle)
            if depth_img is not None:
                depth_img = T.functional.rotate(depth_img, angle, interpolation=T.InterpolationMode.NEAREST)
            if mask_img is not None:
                mask_img = T.functional.rotate(mask_img, angle, interpolation=T.InterpolationMode.NEAREST)
        return rgb_img, depth_img, mask_img



class Random_Crop:
    def __init__(self, min_crop_size, max_crop_size):
        # min_crop_size und max_crop_size sind Tupel (min_h, min_w), (max_h, max_w)
        self.min_crop_size = min_crop_size
        self.max_crop_size = max_crop_size
    
    def __call__(self, images):
        rgb_img, depth_img, mask_img = images
        
        if random.random() > 0.7:
            # random crop-size
            random_h = random.randint(self.min_crop_size[0], self.max_crop_size[0])
            random_w = random.randint(self.min_crop_size[1], self.max_crop_size[1])
            random_crop_size = (random_h, random_w)

            # get random parameters for cropping
            i, j, h, w = T.RandomCrop.get_params(rgb_img, output_size=random_crop_size)
            
            # crop for all images
            rgb_img = T.functional.crop(rgb_img, i, j, h, w)
            if depth_img is not None:
                depth_img = T.functional.crop(depth_img, i, j, h, w)
            if mask_img is not None:
                mask_img = T.functional.crop(mask_img, i, j, h, w)
            
        return rgb_img, depth_img, mask_img



class Random_Brightness_Contrast:
    def __init__(self, brightness_range=0.2, contrast_range=0.2):
        self.brightness = brightness_range
        self.contrast = contrast_range

    def __call__(self, images):
        rgb_img, depth_img, mask_img = images
        
        if random.random() > 0.6:
            brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
            contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
            rgb_img = T.functional.adjust_brightness(rgb_img, brightness_factor)
            rgb_img = T.functional.adjust_contrast(rgb_img, contrast_factor)
        
        return rgb_img, depth_img, mask_img



class Add_Gaussian_Noise:
    def __init__(self, mean=0, std=0.01):
        self.mean = mean
        self.std = std
    
    def __call__(self, images):
        rgb_img, depth_img, mask_img = images

        # Convert Pillow image to NumPy array
        rgb_img_np = np.array(rgb_img)
        
        # Apply Gaussian noise to RGB image
        if random.random() > 0.6:
            noise_rgb = np.random.randn(*rgb_img_np.shape) * self.std + self.mean
            rgb_img_np = np.clip(rgb_img_np + noise_rgb, 0, 255).astype(np.uint8)
        
        # Convert back to Pillow image
        rgb_img = Image.fromarray(rgb_img_np)

        # Handle depth image if it's not None
        if random.random() > 0.6 and depth_img is not None:
            depth_img_np = np.array(depth_img)
            noise_depth = np.random.randn(*depth_img_np.shape) * self.std + self.mean
            depth_img_np = np.clip(depth_img_np + noise_depth, 0, 255).astype(np.uint8)
            depth_img = Image.fromarray(depth_img_np)

        return rgb_img, depth_img, mask_img


class Random_Gaussian_Blur:
    def __init__(self, kernel_size=5, sigma=(0.1, 2.0)):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, images):
        rgb_img, depth_img, mask_img = images
        
        if random.random() > 0.95:
            sigma = random.uniform(*self.sigma)
            rgb_img = T.GaussianBlur(kernel_size=self.kernel_size, sigma=sigma)(rgb_img)
            if depth_img is not None:
                depth_img = T.GaussianBlur(kernel_size=self.kernel_size, sigma=sigma)(depth_img)
        
        return rgb_img, depth_img, mask_img



class Random_Scale:
    def __init__(self, scale_range=(0.8, 1.2)):
        self.scale_range = scale_range
    
    def __call__(self, images):
        rgb_img, depth_img, mask_img = images
        
        if random.random() > 0.6:
            scale_factor = random.uniform(self.scale_range[0], self.scale_range[1])
            h, w = rgb_img.size[-2:]
            new_size = (int(h * scale_factor), int(w * scale_factor))
            rgb_img = T.functional.resize(rgb_img, new_size)
            if depth_img is not None:
                depth_img = T.functional.resize(depth_img, new_size, interpolation=T.InterpolationMode.NEAREST)
            if mask_img is not None:
                mask_img = T.functional.resize(mask_img, new_size, interpolation=T.InterpolationMode.NEAREST)
        
        return rgb_img, depth_img, mask_img



class Random_Background_Modification:
    def __init__(self, bg_value=1, width=1920, height=1080):
        self.bg_value = bg_value
        self.width = width
        self.height = height
    
    def __call__(self, images):
        rgb_img, depth_img, mask_img = images
        rgb_img, depth_img, mask_img = pil_to_cv2([rgb_img, depth_img, mask_img])
        
        if random.random() > 0.6:
            mode = random.choice(["noise", "checkerboard", "gradient pattern", "color shift"])

            if mode == "noise":
                background_pattern = np.random.randint(0, 256, (self.height, self.width, 3), dtype=np.uint8)
            elif mode == "checkerboard":
                checker_size = random.choice([5, 10, 25, 50])
                color1 = random.randint(180, 255)
                color1 = [color1, color1, color1]    # Brighter Color
                color2 = random.randint(0, 130)
                color2 = [color2, color2, color2]    # Darker Color

                # Create the checkerboard pattern
                background_pattern = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                for i in range(0, self.height, checker_size):
                    for j in range(0, self.width, checker_size):
                        color = color1 if (i // checker_size + j // checker_size) % 2 == 0 else color2
                        background_pattern[i:i+checker_size, j:j+checker_size] = color
            elif mode == "gradient pattern":
                background_pattern = np.zeros((self.height, self.width, 3), dtype=np.uint8)

                # Generate a gradient
                if random.random() > 0.5:
                    for i in range(self.height):
                        color_value = int(255 * (i / self.height))
                        background_pattern[i, :] = [color_value, color_value, color_value]
                else:
                    for i in range(self.width):
                        color_value = int(255 * (i / self.width))
                        background_pattern[:, i] = [color_value, color_value, color_value]
            else:
                B, G, R = cv2.split(rgb_img)

                # create shift
                add_B = np.full(B.shape, random.randint(10, 150), dtype=np.uint8)
                add_G = np.full(G.shape, random.randint(10, 150), dtype=np.uint8)
                add_R = np.full(R.shape, random.randint(10, 150), dtype=np.uint8)
                
                # make shift
                shifted_B = cv2.add(B, add_B) if random.random() > 0.5 else cv2.subtract(B, add_B)

                shifted_G = cv2.add(G, add_G) if random.random() > 0.5 else cv2.subtract(G, add_G)

                shifted_R = cv2.add(R, add_R) if random.random() > 0.5 else cv2.subtract(R, add_R)

                # apply shift
                background_pattern = cv2.merge((shifted_B, shifted_G, shifted_R))

            # apply pattern only on background:

            # get pattern in right size
            background_pattern = cv2.resize(background_pattern, (rgb_img.shape[1], rgb_img.shape[0]))

            # Create mask for background and objects
            bg_mask = (mask_img == self.bg_value).astype(np.uint8)
            fg_mask = 1 - bg_mask

            # Combine the original image and generated pattern
            background_with_pattern = cv2.bitwise_and(background_pattern, background_pattern, mask=bg_mask)
            objects_only = cv2.bitwise_and(rgb_img, rgb_img, mask=fg_mask)

            # Overlay the generated pattern and the original objects
            result = cv2.add(background_with_pattern, objects_only)
        else:
            result = rgb_img
        
        # Convert back to cv2
        result, depth_img, mask_img = cv2_to_pil([result, depth_img, mask_img])
        return result, depth_img, mask_img



class Resize:
    def __init__(self, width=1920, height=1080):
        self.target_size = (height, width)

    def __call__(self, images):
        rgb_img, depth_img, mask_img = images
        
        resize_transform = T.Resize(self.target_size)
        
        # resize rgb image
        width, height  = rgb_img.size
        if (height, width) != self.target_size:
            rgb_img = resize_transform(rgb_img)
        
        # resize depth data
        if depth_img is not None:
            width, height = depth_img.size
            if (height, width) != self.target_size:
                depth_img = resize_transform(depth_img)
              
        # resize mask  
        if mask_img is not None:
            width, height  = mask_img.size
            if (height, width) != self.target_size:
                mask_img = resize_transform(mask_img)
        
        return rgb_img, depth_img, mask_img



class Train_Augmentations:
    def __init__(self, width, height,
                 apply_random_flip, apply_random_rotation,
                 apply_random_crop, apply_random_brightness_contrast,
                 apply_random_gaussian_noise, apply_random_gaussian_blur,
                 apply_random_scale,
                 apply_random_background_modification,
                 log_path=None, should_log=True, should_print=True):
        transformations = []
        
        info_str = "Using following Data Augmentations: "
        
        if apply_random_flip:
            transformations += [Random_Flip()]
            info_str += "Random Flip, "
            
        if apply_random_rotation:
            transformations += [Random_Rotation(max_angle=30)]
            info_str += "Random Rotation, "
            
        if apply_random_crop:
            transformations += [Random_Crop(min_crop_size=(100, 100), max_crop_size=(200, 200))]
            info_str += "Random Crop, "
        
        if apply_random_brightness_contrast:
            transformations += [Random_Brightness_Contrast(brightness_range=0.2, contrast_range=0.2)]
            info_str += "Random Brightness Contrast, "
    
        if apply_random_gaussian_noise:
            transformations += [Add_Gaussian_Noise(mean=0, std=0.01)]
            info_str += "Random Gaussian Noise, "
        
        if apply_random_gaussian_blur:
            transformations += [Random_Gaussian_Blur(kernel_size=5, sigma=(0.1, 2.0))]
            info_str += "Random Gaussian Blur, "
        
        if apply_random_scale:
            transformations += [Random_Scale(scale_range=(0.8, 1.2))]
            info_str += "Random Scale, "

        if apply_random_background_modification:
            # added through the update function
            # transformations += [Random_Background_Modification(bg_value=1, width=width, height=height)]
            info_str += "Random Background Modification, "
            
            
        if len(transformations) <= 0:
            log(log_path, "Using no Data Augmentation!", should_log=should_log, should_print=should_print)
        else:
            log(log_path, info_str[:-2], should_log=should_log, should_print=should_print)
    
        if not apply_random_background_modification:
            transformations += [Resize(width=width, height=height)]
    
        self.width = width
        self.height = height
        self.transformations = transformations
        self.augmentations = T.Compose(transformations)
        self.apply_random_background_modification = apply_random_background_modification

    def update(self, bg_value=None):
        if self.apply_random_background_modification and bg_value is None:
            raise ValueError("If choosing apply_random_background_modification = True,you have to pass a background value to the augmentation update.")

        if self.apply_random_background_modification:
            self.transformations += [Random_Background_Modification(bg_value=bg_value, width=self.width, height=self.height)]

        if self.apply_random_background_modification:
            self.transformations += [Resize(width=self.width, height=self.height)]
    
        self.augmentations = T.Compose(self.transformations)

    def __call__(self, rgb_img, depth_img, mask_img):
        
        # Convert to Pillow
        rgb_img, depth_img, mask_img = cv2_to_pil([rgb_img, depth_img, mask_img])
        
        # Apply Transformations/Augmentations
        rgb_img, depth_img, mask_img = self.augmentations((rgb_img, depth_img, mask_img))
        
        # Convert back to cv2
        rgb_img, depth_img, mask_img = pil_to_cv2([rgb_img, depth_img, mask_img])
        
        return rgb_img, depth_img, mask_img



class Inference_Augmentations:
    def __init__(self, width, height):
        self.augmentations = T.Compose([
            Resize(width=width, height=height),
        ])

    def update(self, bg_value=None):
        pass

    def __call__(self, rgb_img, depth_img, mask_img):
        
        # Convert to Pillow
        rgb_img, depth_img, mask_img = cv2_to_pil([rgb_img, depth_img, mask_img])
        
        # Apply Transformations/Augmentations
        rgb_img, depth_img, mask_img = self.augmentations((rgb_img, depth_img, mask_img))
        
        # Convert back to cv2
        rgb_img, depth_img, mask_img = pil_to_cv2([rgb_img, depth_img, mask_img])
        
        return rgb_img, depth_img, mask_img



def cv2_to_pil(images:list):
    result = []
    for image in images:
        if image is None:
            result.append(None)
        elif isinstance(image, np.ndarray):
            result.append(Image.fromarray(image))
        else:
            result.append(image)
            
    return result



def pil_to_cv2(images:list):
    result = []
    for image in images:
        if image is None:
            result.append(None)
        else:
            result.append(np.array(image))
            
    return result






############
# training #
############

def log(file_path, content, reset_logs=False, should_log=True, should_print=True):
    """
    Logs content to a specified file and optionally prints it to the console.

    This function handles logging by writing content to a file at the specified 
    file path. If the file does not exist or if the `reset_logs` flag is set to 
    True, it will create a new log file or clear the existing one. The function 
    also offers an option to print the content to the console.

    Parameters:
    -----------
    file_path (str or None): The path to the log file. If None, the function 
                              does nothing.
    content (str): The content to be logged. This will be appended to the file.
    reset_logs (bool): If True, the log file will be cleared before writing 
                       the new content. Default is False.
    should_print (bool): If True, the content will also be printed to the console. 
                         Default is True.

    Returns:
    --------
    None

    Example:
    --------
    >>> log("logs/my_log.txt", "This is a log entry.")
    
    Note:
    -----
    Ensure that the directory for the log file exists; this function does not 
    create intermediate directories.
    """
    if file_path is None:
        return
    
    if should_log:
        if not os.path.exists(file_path) or reset_logs:
            os.makedirs("/".join(file_path.split("/")[:-1]), exist_ok=True)
            with open(file_path, "w") as f:
                f.write("")

        with open(file_path, "a") as f:
            f.write(f"\n{content}")

    if should_print:
        print(content)



def update_output(cur_epoch,
                    cur_iteration, max_iterations,
                    data_size,
                    duration, 
                    eta_str,
                    total_loss,
                    losses,
                    batch_size,
                    log_path):
    """
    Updates and logs the training output for the YOLACT model.

    This function formats and prints the current training status, including
    epoch and iteration details, duration, estimated time of arrival (ETA),
    total loss, and specific loss metrics. It also logs the output to a
    specified log file.

    Parameters:
    -----------
    cur_epoch (int): Current epoch number.
    cur_iteration (int): Current iteration number.
    max_iterations (int): Total number of iterations for the training.
    data_size (int): Total size of the training dataset.
    duration (float): Duration of the current iteration in seconds.
    eta_str (str): Estimated time of arrival as a string.
    total_loss (float): Total loss for the current iteration.
    losses (dict): Dictionary of specific losses to be displayed.
    batch_size (int): Size of the batch used in training.
    log_path (str): Path to the log file where output should be written.

    Returns:
    --------
    None
    """
    now = datetime.now()
    output = f"Mask-RCNN Training - {now.hour:02}:{now.minute:02} {now.day:02}.{now.month:02}.{now.year:04}"

    detail_output = f"\n| epoch: {cur_epoch:>5} || iteration: {cur_iteration:>8} || duration: {duration:>8.3f} || ETA: {eta_str:>8} || total loss: {total_loss:>8.3f} || "
    detail_output += ''.join([f' {key}: {value:>8.3f} |' for key, value in losses])

    iterations_in_cur_epoch = cur_iteration - cur_epoch*(data_size // batch_size)
    cur_epoch_progress =  iterations_in_cur_epoch / max(1, data_size // batch_size)
    cur_epoch_progress = min(int((cur_epoch_progress*100)//10), 10)
    cur_epoch_progress_ = max(10-cur_epoch_progress, 0)

    cur_total_progress = cur_iteration / max_iterations
    cur_total_progress = min(int((cur_total_progress*100)//10), 10)
    cur_total_progress_ = max(10-cur_total_progress, 0)

    percentage_output = f"\nTotal Progress: |{'#'*cur_total_progress}{' '*cur_total_progress_}|    Epoch Progress: |{'#'*cur_epoch_progress}{' '*cur_epoch_progress_}|"

    print_output = f"\n\n{'-'*32}\n{output}\n{detail_output}\n{percentage_output}\n"


    # print new output
    clear_printing()

    log(log_path, print_output)



def pad_masks(masks, max_num_objs):
    # Amount of masks/objects
    num_objs, height, width = masks.shape
    # Add empty masks so that every datapoint in the current batch have the same amount
    padded_masks = torch.zeros((max_num_objs, height, width), dtype=torch.uint8)
    padded_masks[:num_objs, :, :] = masks  # Add original masks
    return padded_masks



def train_loop(log_path, learning_rate, momentum, decay, num_epochs, 
                batch_size, dataset, data_loader, name, experiment_tracking,
                use_depth, weights_path, should_log=True, should_save=True,
                return_objective='model', mask_score_threshold=0.9,
                calc_metrics=False):
    """
    Train the Mask R-CNN model with the specified parameters.

    Parameters:
    -----------
    log_path (str): Path to the log file.
    learning_rate (float): Learning rate for the optimizer.
    momentum (float): Momentum for the optimizer.
    decay (float): Weight decay for the optimizer.
    num_epochs (int): Number of epochs for training.
    batch_size (int): Size of the batch.
    dataset: The dataset used for training.
    data_loader: Data loader for the training data.
    name (str): Name of the model for saving.
    experiment_tracking (bool): Whether to track experiments.
    use_depth (bool): Whether to use depth information.
    weights_path (str): Path to the pre-trained weights.
    should_log (bool): Should there be prints and logs?
    should_save (bool): Should the model get saved?
    return_objective (str): Decides the return value -> 'loss', 'model'

    Returns:
    --------
    None
    """

    # Create Mask-RCNN with Feature Pyramid Network (FPN) as backbone
    if should_log:
        log(log_path, "Create the model and preparing for training...")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = load_maskrcnn(weights_path=weights_path, use_4_channels=use_depth, pretrained=False, log_path=log_path, should_log=should_log, should_print=should_log)
    model = model.to(device)

    # Optimizer
    # params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
    # optimizer = Adam([
    #                     {'params': model.backbone.parameters(), 'lr': 1e-5},  # Backbone with small learnrate
    #                     {'params': model.rpn.parameters(), 'lr': 1e-4},       # RPN 
    #                     {'params': model.roi_heads.parameters(), 'lr': 1e-3},  # ROI Heads
    #                 ], 
    #                  lr=learning_rate, 
    #                  weight_decay=decay)    # , momentum=momentum
    # scheduler = OneCycleLR(optimizer=optimizer, max_lr=0.001, steps_per_epoch=len(dataset), epochs=num_epochs)
    scheduler = CyclicLR(optimizer=optimizer, base_lr=learning_rate, max_lr=9e-5, step_size_up=int((len(dataset)/batch_size)/2))

    # Experiment Tracking
    if experiment_tracking:
        # Add model for tracking
        # batch = next(iter(data_loader))
        # images, target, _ = batch
        # single_example = torch.zeros_like(images[0]).tolist()
        mlflow.pytorch.log_model(model, "Mask R-CNN")    # , conda_env="./conda_env.yml")    # , input_example=single_example)

        # Init TensorBoard writer
        writer = SummaryWriter()

    # Init
    data_size = len(dataset)
    iteration = 0
    max_iterations = int( (data_size/batch_size)*num_epochs )
    last_time = time.time()
    times = []
    loss_avgs = dict()
    if calc_metrics:
        eval_sum_dict = dict()
        eval_str = ""
        learnrate_str = ""

    # Training
    if should_log:
        log(log_path, "Training starts now...")
    model.train()
    try:
        for epoch in range(num_epochs):
            for images, targets, names in data_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()    # gradient to zero
                losses.backward()        # create backward gradients
                optimizer.step()         # adjust weights towards gradients
                scheduler.step()         # adjust learnrate
                
                if calc_metrics:
                    # Calc Metrics
                    try:
                        model = model.eval()
                        with torch.no_grad():
                            outputs = model(images)
                        for output, target in zip(outputs, targets):
                            # print(f"Mask-GT: {target['masks'].cpu().numpy().shape}")
                            # print(f"Results-Mask: {output['masks'].cpu().numpy().shape}")
                            
                            masks_gt = target['masks'].cpu().numpy()
                            masks_gt = np.transpose(masks_gt, (1, 2, 0))
                            extracted_masks_gt = extract_and_visualize_mask(masks_gt, image=None, ax=None, visualize=False, color_map=None, soft_join=False)
                            
                            result_masks = output['masks'].cpu().squeeze(1).numpy()
                            result_masks = (result_masks > mask_score_threshold).astype(np.uint8)
                            result_masks = np.transpose(result_masks, (1, 2, 0))
                            extracted_result_mask = extract_and_visualize_mask(result_masks, image=None, ax=None, visualize=False, color_map=None, soft_join=False)
                            
                            eval_results = eval_pred(extracted_result_mask, extracted_masks_gt, name="train", should_print=False, should_save=False, save_path=None)
                            eval_sum_dict = update_evaluation_summary(sum_dict=eval_sum_dict, results=eval_results)
                            
                            eval_str = "\nMetrics:"
                            for key, value in eval_sum_dict.items():
                                value = round(statistics.mean(value), 6)
                                
                                eval_str += f"\n    -> {key}: {round(value, 4)}"
                                
                                if experiment_tracking:
                                    mlflow.log_metric(key, value, step=iteration)
                                    writer.add_scalar(key, value, iteration)
                    except Exception as e:
                        log(log_path, f"Error Occured during Metrics calculation: {e}", should_log=should_log, should_print=should_log)
                    model = model.train()
                    

                # get current learnrates
                learnrate_str = "\nLearnrate:"
                if len(optimizer.param_groups) > 1:
                    for param_group in optimizer.param_groups:
                        # get group name
                        parameter_name = None
                        for param in param_group['params']:
                            for name, p in model.named_parameters():
                                if p is param:
                                    parameter_name = name.split(".")[0].upper()
                                    break
                            if parameter_name is not None:
                                break
                        if parameter_name is None:
                            parameter_name = "Unknown"
                        learnrate_str += f"\n    - {parameter_name}: {param_group['lr']:.0e}"
                    
                    mlflow.log_metric(f"learnrate_{parameter_name}", param_group['lr'], step=iteration)
                    writer.add_scalar(f"learnrate_{parameter_name}", param_group['lr'], iteration)
                else:
                    # current_learnrate = scheduler.get_lst_lr()[0]
                    current_learnrate = optimizer.param_groups[0]['lr']
                    learnrate_str += f" {current_learnrate:.0e}"
                    mlflow.log_metric(f"learnrate", current_learnrate, step=iteration)
                    writer.add_scalar(f"learnrate", current_learnrate, iteration)
                    
                # log loss avg
                for key, value in loss_dict.items():
                    if experiment_tracking:
                        # make experiment tracking
                        mlflow.log_metric(key, value.cpu().detach().numpy(), step=iteration)

                        # make tensorboard logging
                        writer.add_scalar(key, value.cpu().detach().numpy(), iteration)

                    if key in loss_avgs.keys():
                        loss_avgs[key] += [value.cpu().detach().numpy()]
                    else:
                        loss_avgs[key] = [value.cpu().detach().numpy()]

                cur_total_loss = sum([value.cpu().detach().numpy() for value in loss_dict.values()])
                
                if experiment_tracking:

                    # make experiment tracking
                    mlflow.log_metric("total loss", cur_total_loss, step=iteration)

                    # make tensorboard logging
                    writer.add_scalar("total loss", cur_total_loss, iteration)

                # log time duration
                cur_time = time.time()
                duration = cur_time - last_time
                last_time = cur_time
                times += [duration]

                if iteration % 10 == 0:
                    
                    eta_str = str(timedelta(seconds=(max_iterations-iteration) * np.mean(np.array(times)))).split('.')[0]
                        
                    total_loss = sum([np.mean(np.array(loss_avgs[k])) for k in loss_avgs.keys()])
                    loss_labels = [[key, np.mean(np.array(value))] for key, value in loss_avgs.items()]

                    # log & print info
                    if should_log:
                        update_output(
                            cur_epoch=epoch,
                            cur_iteration=iteration, 
                            max_iterations=max_iterations,
                            duration=duration,
                            eta_str=eta_str,
                            data_size=data_size,
                            total_loss=total_loss,
                            losses=loss_labels,
                            batch_size=batch_size,
                            log_path=log_path
                        )
                        if calc_metrics:
                            log(log_path, eval_str, should_log=should_log, should_print=should_log)

                        log(log_path, learnrate_str, should_log=should_log, should_print=should_log)

                    # reset
                    times = []
                    loss_avgs = dict()
                    if calc_metrics:
                        eval_sum_dict = dict()

                iteration += 1
                # torch.cuda.empty_cache()

            # Save Model
            if should_save and epoch % 5 == 0:
                torch.save(model.state_dict(), f'./weights/{name}_epoch_{epoch:03}.pth')
    except KeyboardInterrupt:
        if should_log:
            log(log_path, "\nStopping early. Saving network...")
        if should_save:
            torch.save(model.state_dict(), f'./weights/{name}.pth')

        if experiment_tracking:
            try:
                writer.close()
            except Exception:
                pass
            return

    if experiment_tracking:
        try:
            writer.close()
        except Exception:
            pass
        return

    # log & print info
    if should_log:
        update_output(
            cur_epoch=num_epochs,
            cur_iteration=iteration, 
            max_iterations=max_iterations,
            duration=duration,
            eta_str=eta_str,
            data_size=data_size,
            total_loss=total_loss,
            losses=loss_labels,
            batch_size=batch_size,
            log_path=log_path
        )
        if calc_metrics:
            log(log_path, eval_str, should_log=should_log, should_print=should_log)

        log(log_path, f"\nCongratulations!!!! Your Model trained succefull!\n\n Your model waits here for you: '{f'./weights/{name}.pth'}'", should_log=True, should_print=True)

    if return_objective.lower() == "loss":
        return cur_total_loss
    elif return_objective.lower() == "model":
        return model
    else:
        return



def train(
        name='mask_rcnn',
        weights_path=None,
        num_epochs=20,
        learning_rate=0.005,
        momentum=0.9,
        decay=0.0005,
        batch_size = 2,
        img_dir='/home/local-admin/data/3xM/3xM_Dataset_1_1_TEST/rgb',
        depth_dir='/home/local-admin/data/3xM/3xM_Dataset_1_1_TEST/depth-prep',
        mask_dir = '/home/local-admin/data/3xM/3xM_Dataset_1_1_TEST/mask-prep',
        num_workers=4,
        shuffle=True,
        amount=100,     # for random mode
        start_idx=0,    # for range mode
        end_idx=99,     # for range mode
        image_name="3xM_0_10_10.jpg", # for single mode
        data_mode=DATA_LOADING_MODE.ALL,
        use_depth=False,
        using_experiment_tracking=False,
        create_new_experiment=True,    
        # experiment_id=12345,
        experiment_name='3xM Instance Segmentation',
        width=1920,
        height=1080,
        apply_random_flip=True, 
        apply_random_rotation=True,
        apply_random_crop=True, 
        apply_random_brightness_contrast=True,
        apply_random_gaussian_noise=True, 
        apply_random_gaussian_blur=True,
        apply_random_scale=True,
        apply_random_background_modification=True,
        mask_score_threshold=0.9,
        verify_data=True
    ):
    """
    Trains a Mask R-CNN model for instance segmentation using PyTorch.

    Parameters:
    -----------
    name : str, optional
        The name of the model. Default is 'mask_rcnn'.
    weights_path : str or None, optional
        Path to the pre-trained weights. If None, the model will be trained from scratch. Default is None.
    num_epochs : int, optional
        Number of epochs for training. Default is 20.
    learning_rate : float, optional
        The learning rate for the optimizer. Default is 0.005.
    momentum : float, optional
        Momentum factor for the optimizer. Default is 0.9.
    decay : float, optional
        Weight decay (L2 regularization) factor. Default is 0.0005.
    batch_size : int, optional
        Batch size for data loading. Default is 2.
    img_dir : str, optional
        Directory path to the RGB images. Default is '/home/local-admin/data/3xM/3xM_Dataset_1_1_TEST/rgb'.
    depth_dir : str, optional
        Directory path to the depth images. Default is '/home/local-admin/data/3xM/3xM_Dataset_1_1_TEST/depth-prep'.
    mask_dir : str, optional
        Directory path to the mask images. Default is '/home/local-admin/data/3xM/3xM_Dataset_1_1_TEST/mask-prep'.
    num_workers : int, optional
        Number of worker threads for data loading. Default is 4.
    shuffle : bool, optional
        Whether to shuffle the dataset during training. Default is True.
    amount : int, optional
        Number of images to use in random mode. Default is 100.
    start_idx : int, optional
        Starting index for range mode. Default is 0.
    end_idx : int, optional
        Ending index for range mode. Default is 99.
    image_name : str, optional
        Name of the image to load for single mode. Default is '3xM_0_10_10.jpg'.
    data_mode : DATA_LOADING_MODE, optional
        The mode in which to load data. Default is DATA_LOADING_MODE.ALL.
    use_depth : bool, optional
        Whether to include depth data in the training. Default is False.
    using_experiment_tracking : bool, optional
        Whether to use MLflow for experiment tracking. Default is False.
    create_new_experiment : bool, optional
        If True, creates a new experiment in MLflow. Default is True.
    experiment_name : str, optional
        Name of the MLflow experiment. Default is '3xM Instance Segmentation'.
    width : int, optional
        Width of the input images for augmentation. Default is 1920.
    height : int, optional
        Height of the input images for augmentation. Default is 1080.
    apply_random_flip : bool, optional
        Should the data augmentation random flip should be applied. Defaut is True.
    apply_random_rotation : bool, optional
        Should the data augmentation random rotation should be applied. Defaut is True.
    apply_random_crop : bool, optional
        Should the data augmentation random crop should be applied. Defaut is True.
    apply_random_brightness_contrast : bool, optional
        Should the data augmentation random brightness contrast should be applied. Defaut is True.
    apply_random_gaussian_noise : bool, optional
        Should the data augmentation random gaussian noise should be applied. Defaut is True.
    apply_random_gaussian_blur : bool, optional
        Should the data augmentation random gaussian blur should be applied. Defaut is True.
    apply_random_scale : bool, optional
        Should the data augmentation random scale should be applied. Defaut is True.
    apply_random_background_modification : bool, optional
        Should the data augmentation random background modification should be applied. Default is True. Includes Colorshift, Noise, Checkerboard pattern and color gradient pattern.

    Returns:
    --------
    None
        The function trains the Mask R-CNN model and logs results to the console and optionally to MLflow if enabled.
    
    Notes:
    ------
    - The function supports different modes of data loading: random, range, or single image mode.
    - If using MLflow for experiment tracking, ensure MLflow is properly set up.
    - The function logs training information and parameters to a log file at the specified log_path.

    """
    
    clear_printing()
    
    # create folders
    os.makedirs("./weights", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    log_path = f"./logs/{name}.txt"
    log(log_path, "", reset_logs=True, should_print=False)

    welcome_str = "xX Instance Segmentation with MASK-RCNN and PyTorch Xx"
    white_spaces = int(max(0, len(welcome_str)//2 - (len(name)+6)//2))
    log(log_path, f"\n\n{welcome_str}\n{' '*white_spaces}-> {name} <-\n")

    # Dataset and DataLoader
    log(log_path, "Loading the data...")
    augmentation = Train_Augmentations(width=width, height=height, 
                                        apply_random_flip=apply_random_flip, 
                                        apply_random_rotation=apply_random_rotation,
                                        apply_random_crop=apply_random_crop, 
                                        apply_random_brightness_contrast=apply_random_brightness_contrast,
                                        apply_random_gaussian_noise=apply_random_gaussian_noise, 
                                        apply_random_gaussian_blur=apply_random_gaussian_blur,
                                        apply_random_scale=apply_random_scale,
                                        apply_random_background_modification=apply_random_background_modification,
                                        log_path=log_path,
                                        should_log=True,
                                        should_print=True)
    dataset = Dual_Dir_Dataset(img_dir=img_dir, depth_dir=depth_dir, mask_dir=mask_dir, transform=augmentation, 
                                amount=amount, start_idx=start_idx, end_idx=end_idx, image_name=image_name, 
                                data_mode=data_mode, use_mask=True, use_depth=use_depth, log_path=log_path,
                                width=width, height=height, should_log=True, should_print=True, should_verify=verify_data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)

    # Experiment Tracking
    if using_experiment_tracking:

        if create_new_experiment:
            try:
                EXPERIMENT_ID = mlflow.create_experiment(experiment_name)
                log(log_path, f"Created Experiment '{experiment_name}' ID: {EXPERIMENT_ID}")
            except mlflow.exceptions.MlflowException:
                log(log_path, "WARNING: Please set 'CREATE_NEW_EXPERIMENT' to False!")
            # log(log_path, f"IMPORTANT: You should set now 'CREATE_NEW_EXPERIMENT' to False and 'EXPERIMENT_ID' to {experiment_id}.")

        def is_mlflow_active():
            return mlflow.active_run() is not None

        if is_mlflow_active():
            mlflow.end_run()

        # set logs
        existing_experiment = mlflow.get_experiment_by_name(experiment_name)
        if existing_experiment is None:
            raise ValueError("First you have to create a mlflow experiment, you can go to the Variable Section in this notebook.\
                            There you just set 'CREATE_NEW_EXPERIMENT' to True, and run the code there and follow the isntruction there. it's easy, don't worry.\
                            \nAlternativly you can set 'USING_EXPERIMENT_TRACKING' to False.")
        # experiment_id = existing_experiment.experiment_id
        # log(log_path, f"Current Experiment ID: {experiment_id}")
        log(log_path, f"Loaded Experiment-System: {experiment_name}")

        #mlflow.set_tracking_uri(None)
        mlflow.set_experiment(experiment_name)

        if using_experiment_tracking:
            with mlflow.start_run():
                mlflow.set_tag("mlflow.runName", NAME)

                mlflow.log_param("name", NAME)
                mlflow.log_param("epochs", num_epochs)
                mlflow.log_param("batch_size", batch_size)
                mlflow.log_param("learnrate", learning_rate)
                mlflow.log_param("momentum", momentum)
                mlflow.log_param("decay", decay)

                mlflow.log_param("images_path", img_dir)
                mlflow.log_param("masks_path", mask_dir)
                mlflow.log_param("depth_path", depth_dir)
                mlflow.log_param("img_width", width)
                mlflow.log_param("img_height", height)

                mlflow.log_param("data_shuffle", shuffle)
                mlflow.log_param("data_mode", data_mode.value)
                mlflow.log_param("data_amount", amount)
                mlflow.log_param("start_idx", start_idx)
                mlflow.log_param("end_idx", end_idx)

                mlflow.log_param("train_data_size", len(dataset))
                
                mlflow.log_param("apply_random_flip", apply_random_flip)
                mlflow.log_param("apply_random_rotation", apply_random_rotation)
                mlflow.log_param("apply_random_crop", apply_random_crop)
                mlflow.log_param("apply_random_brightness_contrast", apply_random_brightness_contrast)
                mlflow.log_param("apply_random_gaussian_noise", apply_random_gaussian_noise)
                mlflow.log_param("apply_random_gaussian_blur", apply_random_gaussian_blur)
                mlflow.log_param("apply_random_scale", apply_random_scale)

                mlflow.pytorch.autolog()

                train_loop(log_path=log_path, learning_rate=learning_rate, momentum=momentum, decay=decay, 
                            num_epochs=num_epochs, batch_size=batch_size, dataset=dataset, data_loader=data_loader, 
                            name=name, experiment_tracking=using_experiment_tracking, use_depth=use_depth,
                            weights_path=weights_path, should_log=True, should_save=True,
                            return_objective="None")

                # close experiment tracking
                if is_mlflow_active():
                    mlflow.end_run()
        else:
            train_loop(log_path=log_path, learning_rate=learning_rate, momentum=momentum, decay=decay, 
                            num_epochs=num_epochs, batch_size=batch_size, dataset=dataset, data_loader=data_loader, 
                            name=name, experiment_tracking=using_experiment_tracking, use_depth=use_depth,
                            weights_path=weights_path, should_log=True, should_save=True,
                            return_objective="None")




def hyperparameter_optimization(trial,
                                img_dir='/home/local-admin/data/3xM/3xM_Dataset_1_1_TEST/rgb',
                                depth_dir='/home/local-admin/data/3xM/3xM_Dataset_1_1_TEST/depth-prep',
                                mask_dir='/home/local-admin/data/3xM/3xM_Dataset_1_1_TEST/mask-prep',
                                num_workers=4,
                                batch_size=1,
                                momentum=0.9,
                                decay=0.0005,
                                amount=100,     # for random mode
                                start_idx=0,    # for range mode
                                end_idx=99,     # for range mode
                                image_name="3xM_0_10_10.jpg", # for single mode
                                data_mode=DATA_LOADING_MODE.ALL,
                                use_depth=False,
                                width=1920,
                                height=1080,
                                apply_random_flip=True, 
                                apply_random_rotation=True,
                                apply_random_crop=True, 
                                apply_random_brightness_contrast=True,
                                apply_random_gaussian_noise=True, 
                                apply_random_gaussian_blur=True,
                                apply_random_scale=True,
                                verify_data=True
                            ):
    now = datetime.now()
    print(f"    - Start next trial ({now.hour:02}:{now.minute:02} {now.day:02}.{now.month:02}.{now.year:04})")
    
    
    # Hyperparameters to optimize
    learning_rate = trial.suggest_float('learning_rate', 1e-7, 1e-3, log=True)
    # momentum = trial.suggest_float('momentum', 0.7, 0.99)
    # decay = trial.suggest_float('decay', 1e-5, 1e-1, log=True)
    num_epochs = trial.suggest_int('num_epochs', 2, 20) 
    
    
    augmentation = Train_Augmentations(width=width, height=height,
                                        apply_random_flip=apply_random_flip, 
                                        apply_random_rotation=apply_random_rotation,
                                        apply_random_crop=apply_random_crop, 
                                        apply_random_brightness_contrast=apply_random_brightness_contrast,
                                        apply_random_gaussian_noise=apply_random_gaussian_noise, 
                                        apply_random_gaussian_blur=apply_random_gaussian_blur,
                                        apply_random_scale=apply_random_scale,
                                        log_path=None,
                                        should_log=False,
                                        should_print=False)
    dataset = Dual_Dir_Dataset(img_dir=img_dir, depth_dir=depth_dir, mask_dir=mask_dir, transform=augmentation, 
                                amount=amount, start_idx=start_idx, end_idx=end_idx, image_name=image_name, 
                                data_mode=data_mode, use_mask=True, use_depth=use_depth, log_path=None,
                                width=width, height=height, should_log=True, should_print=True, should_verify=verify_data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)

    # Call training function
    total_loss = train_loop(
                    log_path='./logs/trial_log.txt',
                    learning_rate=learning_rate,
                    momentum=momentum,
                    decay=decay,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    dataset=dataset, 
                    data_loader=data_loader,  
                    name='mask_rcnn_trial',
                    experiment_tracking=False, 
                    use_depth=False,
                    weights_path=None,
                    should_log=False, 
                    should_save=False,
                    return_objective="loss"
                )
    
    if total_loss is None:
        total_loss = 7777.777
    else:
        
        total_loss = round(total_loss, 5)

    return total_loss






#############
# inference #
#############



DNN_INSIGHTS = {}

def register_hook(layer, layer_name):
    def hook(module, input, output):
        DNN_INSIGHTS[layer_name] = output.detach().cpu()
    layer.register_forward_hook(hook)



def register_maskrcnn_hooks(model):
    # register_hook(model.fpn.layer2, "fpn.layer2")
    register_hook(model.fpn, "fpn")
    register_hook(model.rpn, "rpn")
    register_hook(model.roi_heads.box_roi_pool, "roi_aligns")



def visualize_insights(insights:dict, should_save, save_path, name, should_show):
    for layer_name, insight in insights.items():
        # insight = insight[0]
        # n_chnneös = insight.shape[0]
        
        # choose a subset to show
        n_channels = insight.size(1)
        n_images = min(4, n_channels)

        fig, ax = plt.subplots(ncols=1, nrows=n_images, figsize=(20*n_images, 15))
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.1)
        
        for i in range(n_images):
            # plot cur insight
            if n_images > 1:
                ax[i].imshow(insight[i].cpu().numpy(), cmap="viridis")
                ax[i].set_title(f"{layer_name} - Channel: {i+1}")
                ax[i].axis("off")
            else:
                ax.imshow(insight[i].cpu().numpy(), cmap="viridis")
                ax.set_title(f"{layer_name} - Channel: {i+1}")
                ax.axis("off")

        plt.suptitle(f"Feature-Map: {layer_name}")

        if should_save:
            plt.savefig(os.path.join(save_path, f"{name}_insights_{layer_name}.jpg"), dpi=fig.dpi)

        if should_show:
            print("\nShowing Ground Truth Visualization*")
            plt.show()
        else:
            plt.clf()



def transform_mask(mask, one_dimensional=False, input_color_map=None):
    """
    Transform a mask into a visible mask with color coding.

    This function converts a segmentation mask into a color image for visualization.
    Each unique value in the mask is assigned a unique color. Optionally, a predefined 
    colormap can be provided. The output can be either a 1-dimensional or 3-dimensional 
    color image.

    Parameters:
    -----------
    mask : numpy.ndarray
        A 2D array representing the segmentation mask where each unique value corresponds 
        to a different object or class.
    one_dimensional : bool, optional
        If True, the output will be a 1-dimensional color image (grayscale). If False, the 
        output will be a 3-dimensional color image (RGB). Default is False.
    input_color_map : list of tuples or list of int, optional
        A list of predefined colors to be used for the objects. Each element should be a 
        tuple of 3 integers for RGB mode or a single integer for grayscale mode. The number 
        of colors in the colormap must be at least equal to the number of unique values in 
        the mask (excluding the background). Default is None.

    Returns:
    --------
    tuple:
        - numpy.ndarray: The color image representing the mask.
        - list: The colormap used for the transformation.

    Raises:
    -------
    ValueError:
        If the dimensions of the input colormap do not match the specified dimensionality 
        (1D or 3D).
    """
    if one_dimensional:
        dimensions = 1
    else:
        dimensions = 3

    if input_color_map and len(input_color_map[0]) != dimensions:
        raise ValueError(f"Dimension of Input Colormap {len(input_color_map[0])} doesn't fit to used dimensions: {dimensions}")

    color_map = []

    # convert mask map to color image
    color_image = np.zeros((mask.shape[0], mask.shape[1], dimensions), dtype=np.uint8)

    # assign a random color to each object and skip the background
    unique_values = np.unique(mask)
    idx = 0
    for value in unique_values:
        if value != 0:
            if input_color_map:
                color = input_color_map[idx]
            else:
                if dimensions == 3:
                    color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                else:
                    color = (np.random.randint(20, 255))
            color_image[mask == value] = color
            color_map += [color]
            idx += 1
    return color_image, color_map



def extract_and_visualize_mask(masks, image=None, ax=None, visualize=True, color_map=None, soft_join=False):
    """
    Extracts masks from a 3D mask array and optionally visualizes them.

    This function takes a 3D array of masks and combines them into a single 2D mask image.
    Optionally, it can visualize the result by overlaying the mask on an input image and 
    displaying it using matplotlib. It returns the extracted 2D mask, and optionally the 
    colorized mask and the colormap.

    Parameters:
    -----------
    masks : numpy.ndarray
        A 3D array of shape (width, height, num_masks) where each mask corresponds to a 
        different object or class.
    image : numpy.ndarray, optional
        An optional 3D array representing the image on which the mask will be overlaid for 
        visualization. It should be of shape (width, height, 3). Default is None.
    ax : matplotlib.axes.Axes, optional
        A matplotlib Axes object to plot the visualization. If None, a new figure and axes 
        will be created. Default is None.
    visualize : bool, optional
        If True, the function will visualize the mask overlay on the image. Default is True.
    color_map : list of tuples, optional
        A list of predefined colors to be used for the objects in the mask. Each element 
        should be a tuple of 3 integers representing an RGB color. Default is None.
    soft_join : bool, optional
        If True, the mask will be softly blended with the input image. If False, the mask 
        will be directly overlaid on the image. Default is False.

    Returns:
    --------
    numpy.ndarray
        The 2D mask image of shape (width, height) where each unique value corresponds to 
        a different object or class.
    numpy.ndarray, optional
        The colorized mask image of shape (width, height, 3) if `visualize` is True.
    list, optional
        The colormap used for the transformation if `visualize` is True.

    Raises:
    -------
    IndexError
        If there is an error in accessing the mask slices due to incorrect shape or size.
    """
    shape = (masks.shape[0], masks.shape[1], 1)

    # calc mask
    if masks.size == 0:
        result_mask = np.full(shape, 0, np.uint8)
    else:
        try:
            result_mask = np.where(masks[:, :, 0], 1, 0)
        except IndexError as e:
            raise IndexError(f"error in 'result_mask = np.where(masks[:, :, 0], 1, 0)'.\nShape of mask: {masks.shape}\nSize of mask: {masks.size}")
        for idx in range(1, masks.shape[2]):
            cur_mask = np.where(masks[:, :, idx], idx+1, 0)
            result_mask = np.where(cur_mask == 0, result_mask, cur_mask)
        result_mask = result_mask.astype("int")

    # visualize
    if visualize:
        color_image, color_map = transform_mask(result_mask, one_dimensional=False, input_color_map=color_map)

        if image is not None:
            color_image = color_image.astype(int) 

            h, w, c = color_image.shape

            if soft_join == False:
                for cur_row_idx in range(h):
                    for cur_col_idx in range(w):
                        if color_image[cur_row_idx, cur_col_idx].sum() != 0:
                            image[cur_row_idx, cur_col_idx] = color_image[cur_row_idx, cur_col_idx]
                color_image = image
            else:
                # remove black baclground
                for cur_row_idx in range(h):
                    for cur_col_idx in range(w):
                        if color_image[cur_row_idx, cur_col_idx].sum() == 0:
                            color_image[cur_row_idx, cur_col_idx] = image[cur_row_idx, cur_col_idx]

                # Set the transparency levels (alpha and beta)
                alpha = 0.7  # transparency of 1. image
                beta = 1 - alpha  # transparency of 2. image

                # Blend the images
                color_image = cv2.addWeighted(image, alpha, color_image, beta, 0)
                # color_image = cv2.add(color_image, image)

        if ax is None:
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20, 15), sharey=True)
            fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=None)
            
        ax.imshow(color_image, vmin=0, vmax=255)
        ax.set_title("Instance Segmentation Mask")
        ax.axis("off")

        return result_mask, color_image, color_map

    return result_mask



def visualize_results(image, predictions, score_threshold=0.5):
    image = np.copy(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Ensure the image is in uint8 format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    pred_scores = predictions[0]['scores'].cpu().numpy()
    pred_boxes = predictions[0]['boxes'].cpu().numpy()
    pred_masks = predictions[0]['masks'].cpu().numpy()
    for idx, score in enumerate(pred_scores):
        if score > score_threshold:
            # Draw bounding box
            box = pred_boxes[idx].astype(int)
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            # Draw Mask
            mask = pred_masks[idx, 0] > 0.5  # Binarisiere die Maske
            colored_mask = np.zeros_like(image, dtype=np.uint8)
            colored_mask[mask] = [0, 0, 255]  # F�rbe die Maske rot
            image = cv2.addWeighted(image, 1, colored_mask, 0.5, 0)
    
    return image



def calc_metric_with_object_matching(mask_1, mask_2, metric_func):
    """
    Calculates the best fitting metric on the given func.
    
    The objects can have different orders, which is not bad at all and have to match better.
    """
    if mask_1.shape != mask_2.shape:
        raise ValueError(f"Can't calculate the IoU between the 2 masks because of different shapes: {mask_1.shape} and {mask_2.shape}")
    
    labels_1 = np.unique(mask_1)
    labels_2 = np.unique(mask_2)
    
    # Remove the background (0 label)
    labels_1 = labels_1[labels_1 != 0]
    labels_2 = labels_2[labels_2 != 0]
    
    metric_matrix = np.zeros((len(labels_1), len(labels_2)))
    
    # Compute the metric for each pair of labels
    for i, label_1 in enumerate(labels_1):
        for j, label_2 in enumerate(labels_2):
            cur_mask_1 = np.where(mask_1 == label_1, 1, 0)
            cur_mask_2 = np.where(mask_2 == label_2, 1, 0)
            metric_matrix[i, j] = metric_func(cur_mask_1, cur_mask_2)
    
    # Use Hungarian algorithm to maximize total metric func across matched pairs
    row_ind, col_ind = linear_sum_assignment(-metric_matrix)  # maximize IoU
    
    # Calculate mean IoU for matched pairs
    matched_metrics = [metric_matrix[i, j] for i, j in zip(row_ind, col_ind)]
    return np.mean(matched_metrics) if matched_metrics else 0.0



def calc_pixel_accuracy(mask_1, mask_2):
    """
    Calculate the pixel accuracy between two masks.

    Args:
    -----
        mask_1 (np.ndarray): The first mask.
        mask_2 (np.ndarray): The second mask.

    Returns:
    --------
        float: The pixel accuracy between the two masks.

    Raises:
    -------
        ValueError: If the shapes of the masks are different.
    """
    if mask_1.shape != mask_2.shape:
        raise ValueError(f"Can't calculate the pixel accuracy between the 2 masks because of different shapes: {mask_1.shape} and {mask_2.shape}")
    
    matching_pixels = np.sum(mask_1 == mask_2)
    all_pixels = np.prod(mask_1.shape)
    return matching_pixels / all_pixels



def calc_bg_fg_acc_accuracy(mask_1, mask_2):
    """
    Calculate the pixel accuracy from the background and foreground between two masks.

    Args:
    -----
        mask_1 (np.ndarray): The first mask.
        mask_2 (np.ndarray): The second mask.

    Returns:
    --------
        float: The pixel accuracy between the two masks.

    Raises:
    -------
        ValueError: If the shapes of the masks are different.
    """
    if mask_1.shape != mask_2.shape:
        raise ValueError(f"Can't calculate the pixel accuracy between the 2 masks because of different shapes: {mask_1.shape} and {mask_2.shape}")
    
    matching_pixels = np.sum((mask_1 > 0) & (mask_2 > 0)) + np.sum((mask_1 == 0) & (mask_2 == 0))
    all_pixels = np.prod(mask_1.shape)
    return matching_pixels / all_pixels
   
    
    
def calc_intersection_over_union(mask_1, mask_2):
    """
    Calculate the Intersection over Union (IoU) between two masks.

    Args:
    -----
        mask_1 (np.ndarray): The first mask.
        mask_2 (np.ndarray): The second mask.

    Returns:
    --------
        float: The IoU between the two masks.

    Raises:
    -------
        ValueError: If the shapes of the masks are different.
    """
    intersection = np.logical_and(mask_1, mask_2)
    union = np.logical_or(mask_1, mask_2)
    
    intersection_area = np.sum(intersection)
    union_area = np.sum(union)
    
    # Avoid division by zero
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area



def calc_precision_and_recall(mask_1, mask_2, only_bg_and_fg=False, aggregation="mean"):
    """
    Calculate the precision and recall between two masks.

    Args:
    -----
        mask_1 (np.ndarray): The first mask.
        mask_2 (np.ndarray): The second mask.
        only_bg_and_fg (bool): Whether to calculate only for background and foreground. Defaults to False.
        aggregation (str): Method to aggregate precision and recall values. Options are "sum", "mean", "median", "std", "var". Defaults to "mean".

    Returns:
    --------
        tuple: Precision and recall values.

    Raises:
    -------
        ValueError: If the shapes of the masks are different.
    """
    if mask_1.shape != mask_2.shape:
        raise ValueError(f"Can't calculate precision and recall between the 2 masks because of different shapes: {mask_1.shape} and {mask_2.shape}")
    
    if only_bg_and_fg:
        TP = np.sum((mask_1 > 0) & (mask_2 > 0))
        FP = np.sum((mask_1 > 0) & (mask_2 == 0))
        FN = np.sum((mask_1 == 0) & (mask_2 > 0))

        precision = TP / (TP + FP) if TP + FP != 0 else 0
        recall = TP / (TP + FN) if TP + FN != 0 else 0
    else:
        precision = []
        recall = []
        unique_labels = np.unique(np.concatenate((mask_1.flatten(), mask_2.flatten())))
        for class_label in unique_labels:
            if class_label != 0:
                TP = np.sum((mask_1 == class_label) & (mask_2 == class_label))
                FP = np.sum((mask_1 == class_label) & (mask_2 != class_label))
                FN = np.sum((mask_1 != class_label) & (mask_2 == class_label))

                precision.append(TP / (TP + FP) if TP + FP != 0 else 0)
                recall.append(TP / (TP + FN) if TP + FN != 0 else 0)

        if aggregation.lower() == "sum":
            precision = np.sum(precision)
            recall = np.sum(recall)
        elif aggregation.lower() in ["avg", "mean"]:
            precision = np.mean(precision)
            recall = np.mean(recall)
        elif aggregation.lower() == "median":
            precision = np.median(precision)
            recall = np.median(recall)
        elif aggregation.lower() == "std":
            precision = np.std(precision)
            recall = np.std(recall)
        elif aggregation.lower() == "var":
            precision = np.var(precision)
            recall = np.var(recall)
    
    return precision, recall



def calc_dice_coefficient(mask_1, mask_2):
    """
    Calculate the Dice Similarity Coefficient between two masks.

    Args:
    -----
        mask_1 (np.ndarray): The first mask.
        mask_2 (np.ndarray): The second mask.

    Returns:
    --------
        float: The Dice coefficient between the two masks.
    """
    intersection = np.logical_and(mask_1, mask_2)
    dice_score = 2 * np.sum(intersection) / (np.sum(mask_1) + np.sum(mask_2))
    
    return dice_score



def calc_f1_score(precision, recall):
    """
    Calculate the F1 Score based on precision and recall.

    Args:
    -----
        precision (float): The precision value.
        recall (float): The recall value.

    Returns:
    --------
        float: The F1 score.
    """
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)



def calc_false_positive_rate(mask_1, mask_2):
    """
    Calculate the False Positive Rate (FPR) between two masks.

    Args:
    -----
        mask_1 (np.ndarray): The first mask.
        mask_2 (np.ndarray): The second mask.

    Returns:
    --------
        float: The False Positive Rate.
    """
    FP = np.sum((mask_1 > 0) & (mask_2 == 0))
    TN = np.sum((mask_1 == 0) & (mask_2 == 0))
    
    return FP / (FP + TN) if FP + TN != 0 else 0



def calc_false_negative_rate(mask_1, mask_2):
    """
    Calculate the False Negative Rate (FNR) between two masks.

    Args:
    -----
        mask_1 (np.ndarray): The first mask.
        mask_2 (np.ndarray): The second mask.

    Returns:
    --------
        float: The False Negative Rate.
    """
    FN = np.sum((mask_1 == 0) & (mask_2 > 0))
    TP = np.sum((mask_1 > 0) & (mask_2 > 0))
    
    return FN / (FN + TP) if FN + TP != 0 else 0


def plot_and_save_evaluation(pixel_acc, bg_fg_acc, iou, precision, recall, f1_score, dice, fpr, fnr, name="instance_segmentation", should_print=True, should_save=True, save_path="./output"):
    eval_str = "\nEvaluation:\n"
    eval_str += f"    - Pixel Accuracy = {round(pixel_acc * 100, 2)}%\n"
    eval_str += f"    - Background Foreground Accuracy = {round(bg_fg_acc * 100, 2)}%\n"
    eval_str += f"    - IoU = {iou}\n"
    eval_str += f"    - Precision = {round(precision * 100, 2)}%\n        -> How many positive predicted are really positive\n        -> Only BG/FG\n"
    eval_str += f"    - Recall = {round(recall * 100, 2)}%\n        -> How many positive were found\n        -> Only BG/FG\n"
    eval_str += f"    - F1 Score = {round(f1_score * 100, 2)}%\n        -> Harmonic mean of Precision and Recall\n"
    eval_str += f"    - Dice Coefficient = {round(dice * 100, 2)}%\n        -> Measure of overlap between predicted and actual masks\n"
    eval_str += f"    - False Positive Rate (FPR) = {round(fpr * 100, 2)}%\n"
    eval_str += f"    - False Negative Rate (FNR) = {round(fnr * 100, 2)}%\n"

    if should_print:
        print(eval_str)

    if should_save:
        path = os.path.join(save_path, f"{name}_eval.txt")
        with open(path, "w") as eval_file:
            eval_file.write(eval_str)
            
    return eval_str


def eval_pred(pred, ground_truth, name="instance_segmentation", should_print=True, should_save=True, save_path="./output"):
    """
    Evaluate prediction against ground truth by calculating pixel accuracy, IoU, precision, recall, F1-score, Dice coefficient, FPR, and FNR.

    Args:
    -----
        pred (np.ndarray): The predicted mask.
        ground_truth (np.ndarray): The ground truth mask.
        should_print (bool): Whether to print the evaluation results. Defaults to True.

    Returns:
    --------
        tuple: Evaluation metrics including pixel accuracy, IoU, precision, recall, F1 score, Dice coefficient, FPR, and FNR.
    """
    pixel_acc = calc_metric_with_object_matching(pred, ground_truth, calc_pixel_accuracy)
    bg_fg_acc = calc_metric_with_object_matching(pred, ground_truth, calc_bg_fg_acc_accuracy)
    iou = calc_metric_with_object_matching(pred, ground_truth, calc_intersection_over_union)
    precision, recall = calc_precision_and_recall(pred, ground_truth, only_bg_and_fg=True)
    f1_score = calc_f1_score(precision, recall)
    dice = calc_metric_with_object_matching(pred, ground_truth, calc_dice_coefficient)
    fpr = calc_metric_with_object_matching(pred, ground_truth, calc_false_positive_rate)
    fnr = calc_metric_with_object_matching(pred, ground_truth, calc_false_negative_rate)

    plot_and_save_evaluation(pixel_acc, iou, precision, recall, f1_score, dice, fpr, fnr, name=name, should_print=should_print, should_save=should_save, save_path=save_path)

    return pixel_acc, bg_fg_acc, iou, precision, recall, f1_score, dice, fpr, fnr



def update_evaluation_summary(sum_dict, results):
    pixel_acc, bg_fg_acc, iou, precision, recall, f1_score, dice, fpr, fnr = results

    eval_key_value = [
        ["pixel accuracy", pixel_acc],
        ["background foreground accuracy", bg_fg_acc],
        ["intersection over union", iou],
        ["precision", precision], 
        ["recall", recall], 
        ["f1 score", f1_score], 
        ["dice", dice], 
        ["false positive rate", fpr], 
        ["false negative rate", fnr]
    ]

    for key, value in eval_key_value:
        if key in sum_dict.keys():
            sum_dict[key] += [value]
        else:
            sum_dict[key] = [value]

    return sum_dict



def save_and_show_evaluation_summary(sum_dict, name="instance_segmentation", should_print=True, should_save=True, save_path="./output"):

    pixel_acc = statistics.mean(sum_dict["pixel accuracy"])
    bg_fg_acc = statistics.mean(sum_dict["background foreground accuracy"])
    iou = statistics.mean(sum_dict["intersection over union"])
    precision = statistics.mean(sum_dict["precision"])
    recall = statistics.mean(sum_dict["recall"])
    f1_score = statistics.mean(sum_dict["f1 score"])
    dice = statistics.mean(sum_dict["dice"])
    fpr = statistics.mean(sum_dict["false positive rate"])
    fnr = statistics.mean(sum_dict["false negative rate"])

    plot_and_save_evaluation(pixel_acc, bg_fg_acc, iou, precision, recall, f1_score, dice, fpr, fnr, name=name, should_print=should_print, should_save=should_save, save_path=save_path)




def inference(  
        weights_path,
        img_dir='/home/local-admin/data/3xM/3xM_Dataset_1_1_TEST/rgb',
        depth_dir='/home/local-admin/data/3xM/3xM_Dataset_1_1_TEST/depth-prep',
        mask_dir = '/home/local-admin/data/3xM/3xM_Dataset_1_1_TEST/mask-prep',
        amount=100,     # for random mode
        start_idx=0,    # for range mode
        end_idx=99,     # for range mode
        image_name="3xM_0_10_10.jpg", # for single mode
        data_mode=DATA_LOADING_MODE.ALL,
        num_workers=4,
        use_mask=True,
        use_depth=False,
        output_type="jpg",
        output_dir="./output",
        should_visualize=True,
        visualization_dir="./output/visualizations",
        save_visualization=True,
        save_evaluation=True,
        show_visualization=False,
        show_evaluation=False,
        width=1920,
        height=1080,
        mask_threshold=0.9,
        show_insights=False,
        save_insights=True,
        verify_data=True
    ):
    """
    Perform inference using a Mask R-CNN model with the provided dataset and parameters, while also
    supporting visualization and evaluation of the results.

    Args:
    -----
        weights_path (str): Path to the pretrained Mask R-CNN model weights.
        img_dir (str): Directory containing RGB images for inference.
        depth_dir (str): Directory containing depth-prepared images (if applicable).
        mask_dir (str): Directory containing ground-truth mask-prepared images (if applicable).
        amount (int): Number of images to process in random mode.
        start_idx (int): Starting index for range mode.
        end_idx (int): Ending index for range mode.
        image_name (str): Image name for single image inference mode.
        data_mode (Enum): Data loading mode, choosing between all, range, random, or single image modes.
        num_workers (int): Number of workers for data loading.
        use_mask (bool): Whether to use ground-truth masks during inference for evaluation and comparison.
        use_depth (bool): Whether to include depth information for the model inference.
        output_type (str): Format to save the inferred masks, options are ['jpg', 'png', 'npy'].
        output_dir (str): Directory to save the inference results.
        should_visualize (bool): Whether to visualize the inference process.
        visualization_dir (str): Directory to save visualized results.
        save_visualization (bool): Whether to save the visualizations.
        save_evaluation (bool): Whether to save evaluation results.
        show_visualization (bool): Whether to display the visualized results.
        width (int): Image width for resizing during inference.
        height (int): Image height for resizing during inference.

    Returns:
    --------
        None: This function outputs inference masks, saves results, and optionally visualizes or evaluates them.
    
    Notes:
    ------
        - The function loads the Mask R-CNN model with the provided weights and runs inference over the dataset.
        - The masks can be saved in several formats, including 'jpg', 'png', or 'npy'.
        - Ground truth comparisons and visualizations can be enabled to further analyze the results.
        - Evaluation is done when ground-truth masks are provided, comparing prediction accuracy.
    """

    global DNN_INSIGHTS

    print(f"\n\nxX Mask-RCNN Inference Xx")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using {device}")

    dataset = Dual_Dir_Dataset(img_dir=img_dir, depth_dir=depth_dir, mask_dir=mask_dir, transform=Inference_Augmentations(width=width, height=height), 
                                amount=amount, start_idx=start_idx, end_idx=end_idx, image_name=image_name, 
                                data_mode=data_mode, use_mask=use_mask, use_depth=use_depth, log_path=None,
                                width=width, height=height, should_log=False, should_print=True, should_verify=verify_data)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=num_workers, collate_fn=collate_fn)

    all_images_size = len(data_loader)

    model_name = ".".join(weights_path.split("/")[-1].split(".")[:-1])

    eval_sum_dict = dict()

    with torch.no_grad():
        # create model
        model = load_maskrcnn(weights_path=weights_path, use_4_channels=use_depth, pretrained=False, log_path=None, should_log=False, should_print=False)
        model.eval()
        model = model.to(device)

        if show_insights or save_insights:
            register_maskrcnn_hooks(model=model)
        
        idx = 1

        for data in data_loader:
            
            # print progress
            clear_printing()
            print(f"\n\nxX Mask-RCNN Inference Xx")
            print(f"Using {device}\n\n\n")
            print(f"Inference {idx}. image from {all_images_size}\n")
            progress = (idx / all_images_size)
            progress_bar_size = 10
            progress_bar = int(progress * progress_bar_size)
            white_spaces = int(progress_bar_size - progress_bar)
            print(f"[{'#'*progress_bar}{' '*white_spaces}]")

            
            if use_mask:
                image = data[0][0].to(device).unsqueeze(0)
                masks = data[1][0]["masks"]
                masks = masks.to(device)
                name = data[2][0]
            else:
                image = data[0][0]
                name = data[1][0]

            # inference
            all_results = model(image)
            result = all_results[0]
            
            # move every data to cpu and bring to right format CHW to WHC
            image = image.cpu().numpy().squeeze(0)
            image = np.transpose(image, (1, 2, 0))  # Convert to HWC
            
            # Remove 4.th channel if existing
            if image.shape[2] == 4:
                image = image[:, :, :3]
            
            if use_mask:
                masks_gt = masks.cpu().numpy()
                masks_gt = np.transpose(masks_gt, (1, 2, 0))
            
            result_masks = result['masks'].cpu().squeeze(1).numpy()
            result_masks = (result_masks > mask_threshold).astype(np.uint8)
            result_masks = np.transpose(result_masks, (1, 2, 0))
            
            # save mask
            os.makedirs(output_dir, exist_ok=True)

            # result = {key: value.cpu() for key, value in result.items()}
            cleaned_name = model_name + "_" + ".".join(name.split(".")[:-1])

            extracted_mask = extract_and_visualize_mask(result_masks, image=None, ax=None, visualize=False, color_map=None, soft_join=False)
            if output_type in ["numpy", "npy"]:
                np.save(os.path.join(output_dir, f'{cleaned_name}.npy'), extracted_mask)
            else:
                # recommended
                cv2.imwrite(os.path.join(output_dir, f'{cleaned_name}.png'), extracted_mask)

            # plot results
            if should_visualize:
                ncols = 3

                fig, ax = plt.subplots(ncols=ncols, nrows=1, figsize=(20, 15), sharey=True)
                fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.02, hspace=None)
                
                # plot original image
                ax[0].imshow(image)
                ax[0].set_title("Original")
                ax[0].axis("off")

                # plot mask alone
                _, color_image, color_map = extract_and_visualize_mask(result_masks, image=None, ax=ax[1], visualize=True)
                ax[1].set_title("Prediction Mask")
                ax[1].axis("off")

                # plot result
                _, _, _ = extract_and_visualize_mask(result_masks, image=image, ax=ax[2], visualize=True, color_map=color_map)    # color_map)
                ax[2].set_title("Result")
                ax[2].axis("off")

                if save_visualization:
                    os.makedirs(visualization_dir, exist_ok=True)
                    plt.savefig(os.path.join(visualization_dir, f"{cleaned_name}.jpg"), dpi=fig.dpi)


                if show_visualization:
                    print("\nShowing Visualization*")
                    plt.show()
                else:
                    plt.clf()
                    
                # use other visualization
                vis_img = visualize_results(image=image, predictions=all_results, score_threshold=0.5)
                if save_visualization:
                    cv2.imwrite(os.path.join(visualization_dir, f"{cleaned_name}_V2.jpg"), vis_img)
                    
                if show_visualization:
                    plt.imshow(vis_img)

            # eval and plot ground truth comparisson
            if use_mask:
                if show_evaluation:
                    print("Plot result in comparison to ground truth and evaluate with ground truth*")
                # mask = cv2.resize(mask, extracted_mask.shape[1], extracted_mask.shape[0])
                masks_gt = extract_and_visualize_mask(masks_gt, image=None, ax=None, visualize=False, color_map=None, soft_join=False)
                eval_results = eval_pred(extracted_mask, masks_gt, name=cleaned_name, should_print=show_evaluation, should_save=save_evaluation, save_path=output_dir)
                eval_sum_dict = update_evaluation_summary(sum_dict=eval_sum_dict, results=eval_results)

                if should_visualize:
                    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(20, 15), sharey=True)
                    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.1)
                    
                    # plot ground_truth
                    mask_gt, _ = transform_mask(masks_gt, one_dimensional=False)
                    ax[0].imshow(mask_gt)
                    ax[0].set_title("Ground Truth Mask")
                    ax[0].axis("off")

                    # plot prediction mask
                    _, color_image, color_map = extract_and_visualize_mask(result_masks, image=None, ax=ax[1], visualize=True, color_map=color_map)
                    ax[1].set_title("Predicted Mask")
                    ax[1].axis("off")

                    if save_visualization:
                        plt.savefig(os.path.join(visualization_dir, f"{cleaned_name}_ground_truth.jpg"), dpi=fig.dpi)

                    if show_visualization:
                        print("\nShowing Ground Truth Visualization*")
                        plt.show()
                    else:
                        plt.clf()

            if show_insights or save_insights:
                visualize_insights(insights=DNN_INSIGHTS, should_save=save_insights, save_path=visualization_dir, name=cleaned_name, should_show=show_insights)
                        
            idx += 1
            DNN_INSIGHTS = {}

        if use_mask:
            save_and_show_evaluation_summary(eval_sum_dict, name="Complete-Evaluation", should_print=show_evaluation, should_save=save_evaluation, save_path=output_dir)






if __name__ == "__main__":

    if MODE == RUN_MODE.TRAIN:
        if MULTIPLE_DATASETS is not None:
            datasets = []
            for cur_dataset in os.listdir(MULTIPLE_DATASETS):
                if cur_dataset not in SKIP_DATASETS:
                    img_path = os.path.join(MULTIPLE_DATASETS, cur_dataset, "rgb")
                    mask_path = os.path.join(MULTIPLE_DATASETS, cur_dataset, "mask")
                    depth_path = os.path.join(MULTIPLE_DATASETS, cur_dataset, "depth")
                    if os.path.exists(img_path) and os.path.isdir(img_path) and \
                        os.path.exists(mask_path) and os.path.isdir(mask_path) and \
                        (os.path.exists(depth_path) and os.path.isdir(depth_path) or not USE_DEPTH):
                        datasets += [cur_dataset]
        else:
            datasets = [1]

        for cur_dataset in datasets:
            if MULTIPLE_DATASETS is not None:
                name = f"{NAME}_{cur_dataset}"
                img_path = os.path.join(MULTIPLE_DATASETS, cur_dataset, "rgb")
                mask_path = os.path.join(MULTIPLE_DATASETS, cur_dataset, "mask")
                depth_path = os.path.join(MULTIPLE_DATASETS, cur_dataset, "depth")
            else:
                name = NAME
                img_path = IMG_DIR
                depth_path = DEPTH_DIR
                mask_path = MASK_DIR

            train(
                name=name,
                weights_path=WEIGHTS_PATH,
                num_epochs=NUM_EPOCHS,
                learning_rate=LEARNING_RATE,
                # momentum=MOMENTUM,
                decay=DECAY,
                batch_size=BATCH_SIZE,
                img_dir=img_path,
                mask_dir=mask_path,
                depth_dir=depth_path,
                num_workers=NUM_WORKERS,
                shuffle=SHUFFLE,
                amount=AMOUNT,         # for random mode
                start_idx=START_IDX,   # for range mode
                end_idx=END_IDX,       # for range mode
                image_name=IMAGE_NAME, # for single mode
                data_mode=DATA_MODE,
                use_depth=USE_DEPTH,
                using_experiment_tracking=USING_EXPERIMENT_TRACKING,
                create_new_experiment=CREATE_NEW_EXPERIMENT,    
                experiment_name=EXPERIMENT_NAME,
                width=WIDTH,
                height=HEIGHT,
                apply_random_flip=APPLY_RANDOM_FLIP, 
                apply_random_rotation=APPLY_RANDOM_ROTATION,
                apply_random_crop=APPLY_RANDOM_CROP, 
                apply_random_brightness_contrast=APPLY_RANDOM_BRIGHTNESS_CONTRAST,
                apply_random_gaussian_noise=APPLY_RANDOM_GAUSSIAN_NOISE, 
                apply_random_gaussian_blur=APPLY_RANDOM_GAUSSIAN_BLUR,
                apply_random_scale=APPLY_RANDOM_SCALE,
                apply_random_background_modification=APPLY_RANDOM_BACKGROUND_MODIFICATION,
                # mask_score_threshold=MASK_SCORE_THRESHOLD,
                verify_data=VERIFY_DATA
            )
    elif MODE == RUN_MODE.HYPERPARAMETER_TUNING:
        print("Start Hyperparameter optimization...")
        
        # add parameters to function
        partial_optimization_func = partial(hyperparameter_optimization, 
                                            img_dir=IMG_DIR,
                                            depth_dir=DEPTH_DIR,
                                            mask_dir=MASK_DIR,
                                            num_workers=NUM_WORKERS,
                                            batch_size=BATCH_SIZE,
                                            # momentum=MOMENTUM,
                                            decay=DECAY,
                                            amount=AMOUNT,     # for random mode
                                            start_idx=START_IDX,    # for range mode
                                            end_idx=END_IDX,     # for range mode
                                            image_name=IMAGE_NAME, # for single mode
                                            data_mode=DATA_MODE,
                                            use_depth=USE_DEPTH,
                                            width=WIDTH,
                                            height=HEIGHT,
                                            apply_random_flip=APPLY_RANDOM_FLIP, 
                                            apply_random_rotation=APPLY_RANDOM_ROTATION,
                                            apply_random_crop=APPLY_RANDOM_CROP, 
                                            apply_random_brightness_contrast=APPLY_RANDOM_BRIGHTNESS_CONTRAST,
                                            apply_random_gaussian_noise=APPLY_RANDOM_GAUSSIAN_NOISE, 
                                            apply_random_gaussian_blur=APPLY_RANDOM_GAUSSIAN_BLUR,
                                            apply_random_scale=APPLY_RANDOM_SCALE,
                                            apply_random_background_modification=APPLY_RANDOM_BACKGROUND_MODIFICATION,
                                            # mask_score_threshold=MASK_SCORE_THRESHOLD
                                            verify_data=VERIFY_DATA
                                        )
                                    
        study = optuna.create_study(direction='minimize')
        study.optimize(partial_optimization_func, n_trials=20) 

        # Print best hyperparameters
        now = datetime.now()
        print(f"Optimization with Optuna is finish! ({now.hour:02}:{now.minute:02} {now.day:02}.{now.month:02}.{now.year:04})")
    
        result_str = f"Best hyperparameters:\n{study.best_params}\n\n\nBest total loss: {study.best_value}"
        print(result_str)
        with open("./optuna_result.txt", "w") as optuna_file:
            optuna_file.write(result_str)
    elif MODE == RUN_MODE.INFERENCE:
        inference(
                weights_path=WEIGHTS_PATH,
                img_dir=IMG_DIR,
                depth_dir=DEPTH_DIR,
                mask_dir = MASK_DIR,
                amount=AMOUNT,
                start_idx=START_IDX,
                end_idx=END_IDX,
                image_name=IMAGE_NAME,
                data_mode=DATA_MODE,
                num_workers=NUM_WORKERS,
                use_mask=USE_MASK,
                use_depth=USE_DEPTH,
                output_type=OUTPUT_TYPE,
                output_dir=OUTPUT_DIR,
                should_visualize=SHOULD_VISUALIZE,
                visualization_dir=VISUALIZATION_DIR,
                save_visualization=SAVE_VISUALIZATION,
                save_evaluation=SAVE_EVALUATION,
                show_visualization=SHOW_VISUALIZATION,
                show_evaluation=SHOW_EVALUATION,
                width=WIDTH,
                height=HEIGHT,
                mask_threshold=MASK_SCORE_THRESHOLD,
                show_insights=SHOW_INSIGHTS,
                save_insights=SAVE_INSIGHTS,
                verify_data=VERIFY_DATA
        )
    
    

