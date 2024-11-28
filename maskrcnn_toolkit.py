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
    INFERENCE = "inference"






#############
# variables #
#############
# Change these variables to your need

MODE = RUN_MODE.INFERENCE



# -------- #
# TRAINING #
# -------- #
if MODE == RUN_MODE.TRAIN:
    EXTENDED_VERSION = True
    WEIGHTS_PATH = None         # Path to the model weights file
    USE_DEPTH = True           # Whether to include depth information -> as rgb and depth on green channel
    VERIFY_DATA = False         # True is recommended

    GROUND_PATH = "/mnt/morespace/3xM"    # "/mnt/morespace/3xM" "D:/3xM" 
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

    MULTIPLE_DATASETS = None # GROUND_PATH         # Path to folder for training multiple models
    SKIP_DATASETS = ["3xM_Test_Datasets", "3xM_Dataset_10_160"]
    NAME = 'extended_mask_rcnn_rgbd'                 # Name of the model to use

    USING_EXPERIMENT_TRACKING = False   # Enable experiment tracking
    CREATE_NEW_EXPERIMENT = True       # Whether to create a new experiment run
    EXPERIMENT_NAME = "3xM Instance Segmentation"  # Name of the experiment

    NUM_EPOCHS = 100                    # Number of training epochs
    WARM_UP_ITER = 2000
    LEARNING_RATE = 3e-3              # Learning rate for the optimizer
    MOMENTUM = 0.9                     # Momentum for the optimizer
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
    


# --------- #
# INFERENCE #
# --------- #
if MODE == RUN_MODE.INFERENCE:
    EXTENDED_VERSION = False
    WEIGHTS_PATH = "./weights/mask_rcnn_rgbd_3xM_Dataset_160_160_epoch_040.pth"  # Path to the model weights file
    MASK_SCORE_THRESHOLD = 0.5
    USE_DEPTH = True                   # Whether to include depth information -> as rgb and depth on green channel
    VERIFY_DATA = True         # True is recommended

    GROUND_PATH = "D:/3xM/3xM_Test_Dataset/"   # "/mnt/morespace/3xM"    "D:/3xM/3xM_Test_Dataset/3xM_Bias_Experiment"
    DATASET_NAME = "3xM_Bias_Experiment"    #  "3xM_Bias_Experiment", "3xM_Test_Dataset_known_known", "OCID-dataset-prep"
    IMG_DIR = os.path.join(GROUND_PATH, DATASET_NAME, 'rgb')        # Directory for RGB images
    DEPTH_DIR = os.path.join(GROUND_PATH, DATASET_NAME, 'depth')    # Directory for depth-preprocessed images
    MASK_DIR = os.path.join(GROUND_PATH, DATASET_NAME, 'mask')      # Directory for mask-preprocessed images

    DATA_MODE = DATA_LOADING_MODE.ALL  # Mode for loading data -> All, Random, Range, Single Image
    AMOUNT = 10                       # Number of images for random mode
    START_IDX = 0                      # Starting index for range mode
    END_IDX = 10                       # Ending index for range mode
    IMAGE_NAME = "3xM_10000_10_80.png"     # Specific image name for single mode

    NUM_WORKERS = 4                    # Number of workers for data loading

    OUTPUT_DIR = "./output"            # Directory to save output files
    USE_MASK = False                    # Whether to use masks during inference
    SHOULD_SAVE_MASK = False
    OUTPUT_TYPE = "png"                # Output format: 'numpy-array' or 'png'
    SHOULD_VISUALIZE_MASK = True
    SHOULD_VISUALIZE_MASK_AND_IMAGE = True
    SAVE_VISUALIZATION = True          # Save the visualizations to disk
    SHOW_VISUALIZATION = False          # Display the visualizations
    SAVE_EVALUATION = False             # Save the evaluation results
    SHOW_EVALUATION = False             # Display the evaluation results
    SHOW_INSIGHTS = False
    SAVE_INSIGHTS = False

    RESET_OUTPUT = True






###########
# imports #
###########

# basics
import shutil
from datetime import datetime, timedelta
import time
from IPython.display import clear_output
from functools import partial
import math
import statistics
import queue
from collections import OrderedDict
import pickle

# image
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2 
from PIL import Image    # for PyTorch Transformations

# deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import LambdaLR, SequentialLR, LinearLR, CosineAnnealingLR

import torchvision
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models import ResNet50_Weights
from torchvision.models.resnet import resnet50
import torchvision.transforms as T

# experiment tracking
from torch.utils.tensorboard import SummaryWriter
import mlflow
import mlflow.pytorch

# optimization
from scipy.optimize import linear_sum_assignment


###########
# general #
###########



class Extended_ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # add first layers
        self.c1 = nn.Sequential(self.backbone.conv1, self.backbone.bn1, self.backbone.relu)  # C1
        self.c2 = nn.Sequential(self.backbone.maxpool, self.backbone.layer1)  # C2
        self.c3 = self.backbone.layer2  # C3
        self.c4 = self.backbone.layer3  # C4
        self.c5 = self.backbone.layer4  # C5

        # just for Mask R-CNN
        self.conv1 = self.backbone.conv1
        self.bn1 = self.backbone.bn1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        c5 = self.c5(c4)
        return c1, c2, c3, c4, c5
    
    def update_conv1(self, new_conv1):
        self.conv1 = new_conv1  # Replace the old Conv1 Layer with the new one
        self.c1 = nn.Sequential(new_conv1, self.backbone.bn1, self.backbone.relu)
            



class Extended_FPN(nn.Module):
    def __init__(self):
        super().__init__()

        # Inner Blocks: 1x1 Convolutions for Channel-Resizing / Adjustment
        self.inner_blocks = nn.ModuleList([
            nn.Conv2d(64, 256, kernel_size=1),   # for C1 (64 Channels)
            nn.Conv2d(256, 256, kernel_size=1),  # for C2
            nn.Conv2d(512, 256, kernel_size=1),  # for C3
            nn.Conv2d(1024, 256, kernel_size=1), # for C4
            nn.Conv2d(2048, 256, kernel_size=1)  # for C5
        ])

        # Layer Blocks: 3x3 Convolutions for more refined results
        self.layer_blocks = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # for P1
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # for P2
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # for P3
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # for P4
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)   # for P5
        ])

    def forward(self, c1, c2, c3, c4, c5):
        # Extract features from body
        # c1, c2, c3, c4, c5 = self.body(x)

        # build feature pyramid
        p5 = self.inner_blocks[4](c5)
        p4 = self.inner_blocks[3](c4) + F.interpolate(p5, scale_factor=2, mode="nearest")
        p3 = self.inner_blocks[2](c3) + F.interpolate(p4, scale_factor=2, mode="nearest")
        p2 = self.inner_blocks[1](c2) + F.interpolate(p3, scale_factor=2, mode="nearest")
        p1 = self.inner_blocks[0](c1) + F.interpolate(p2, scale_factor=2, mode="nearest")

        # Apply layer blocks
        p5 = self.layer_blocks[4](p5)
        p4 = self.layer_blocks[3](p4)
        p3 = self.layer_blocks[2](p3)
        p2 = self.layer_blocks[1](p2)
        p1 = self.layer_blocks[0](p1)

        return [p1, p2, p3, p4, p5]



class Extended_Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = Extended_ResNet()
        self.fpn = Extended_FPN()
        self.out_channels = 256
        
    def forward(self, x):
        c1, c2, c3, c4, c5 = self.body(x)  # Deine ResNet-Features
        fpn_features = self.fpn(c1, c2, c3, c4, c5)
        return {"p1": fpn_features[0], "p2": fpn_features[1], "p3": fpn_features[2], "p4": fpn_features[3], "p5": fpn_features[4]}



def load_maskrcnn(weights_path=None, use_4_channels=False, pretrained=True,
                  image_mean=[0.485, 0.456, 0.406, 0.5], image_std=[0.229, 0.224, 0.225, 0.5],    # from ImageNet
                  min_size=1080, max_size=1920, log_path=None, should_log=False, should_print=True,
                  extended_version=False):
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
    if extended_version:
        backbone = Extended_Backbone()
        model = MaskRCNN(backbone, num_classes=2)  # 2 Classes (Background + 1 Object)

        if use_4_channels:
            # Change the first Conv2d-Layer for 4 Channels
            in_features = model.backbone.body.conv1.in_channels    # this have to be changed
            out_features = model.backbone.body.conv1.out_channels
            kernel_size = model.backbone.body.conv1.kernel_size
            stride = model.backbone.body.conv1.stride
            padding = model.backbone.body.conv1.padding
            
            # Create new conv layer with 4 channels
            new_conv1 = torch.nn.Conv2d(4, out_features, kernel_size=kernel_size, stride=stride, padding=padding)
            
            # Copy the existing weights from the first 3 Channels
            with torch.no_grad():
                new_conv1.weight[:, :3, :, :] = model.backbone.body.conv1.weight  # Copy old 3 Channels
                new_conv1.weight[:, 3:, :, :] = model.backbone.body.conv1.weight[:, :1, :, :]  # Init new 4.th Channel with the one old channel

            # Update model
            model.backbone.body.update_conv1(new_conv1)
            
            # Modify the transform to handle 4 channels
            model.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)


        out_channels = model.backbone.out_channels
        
        # update RPN
        # modify anchors for smaller objects
        rpn_anchor_generator = AnchorGenerator(
            sizes=(
                (16, 32, 64),        # Sizes for level 0
                (32, 64, 128),       # Sizes for level 1
                (64, 128, 256),      # Sizes for level 2
                (128, 256, 512),     # Sizes for level 3
                (256, 512, 1024),    # Sizes for level 4
            ),
            aspect_ratios=(
                (0.25, 0.5, 1.0),  # Aspect ratios for level 0
                (0.25, 0.5, 1.0),  # Aspect ratios for level 1
                (0.25, 0.5, 1.0),  # Aspect ratios for level 2
                (0.25, 0.5, 1.0),  # Aspect ratios for level 3
                (0.25, 0.5, 1.0),  # Aspect ratios for level 4
            ),
        )
        
        rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])
        rpn_fg_iou_thresh = 0.7
        rpn_bg_iou_thresh = 0.3
        rpn_batch_size_per_image = 256
        rpn_positive_fraction = 0.5
        rpn_pre_nms_top_n = dict(training=2000, testing=1000)
        rpn_post_nms_top_n = dict(training=2000, testing=1000)
        rpn_nms_thresh = 0.7
        score_thresh = 0.0
        rpn = RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=score_thresh,
        )
        model.rpn = rpn
        
        # Update RoI, Box and Mask Head
        box_roi_pool = MultiScaleRoIAlign(featmap_names=["p1", "p2", "p3", "p4", "p5"], output_size=7, sampling_ratio=2)

        resolution = box_roi_pool.output_size[0]
        representation_size = 1024
        box_head = TwoMLPHead(out_channels * resolution**2, representation_size)

        representation_size = 1024
        box_predictor = FastRCNNPredictor(representation_size, 2)
        
        box_fg_iou_thresh = 0.5
        box_bg_iou_thresh = 0.5
        box_batch_size_per_image = 512
        box_positive_fraction = 0.25
        bbox_reg_weights = None
        box_score_thresh = 0.05
        box_nms_thresh = 0.5
        box_detections_per_img = 100

        mask_roi_pool = MultiScaleRoIAlign(featmap_names=["p1", "p2", "p3", "p4", "p5"], output_size=14, sampling_ratio=2)
        mask_layers = (256, 256, 256, 256, 256)
        mask_dilation = 1
        mask_head = MaskRCNNHeads(out_channels, mask_layers, mask_dilation)
        
        mask_predictor_in_channels = 256  # == mask_layers[-1]
        mask_dim_reduced = 256
        mask_predictor = MaskRCNNPredictor(mask_predictor_in_channels, mask_dim_reduced, 2)

        roi_heads = RoIHeads(
            # Box
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img
        )
        roi_heads.mask_roi_pool = mask_roi_pool
        roi_heads.mask_head = mask_head
        roi_heads.mask_predictor = mask_predictor
        
        model.roi_heads = roi_heads
            
        # other parameter adjustments
        # adjust loss weights
        model.rpn.rpn_cls_loss_weight = 1.0
        model.rpn.rpn_bbox_loss_weight = 2.0
        model.roi_heads.mask_loss_weight = 2.0
        model.roi_heads.box_loss_weight = 1.0
        model.roi_heads.classification_loss_weight = 1.0
        
        # adjust non-maximum suppression
        model.roi_heads.nms_thresh = 0.2
        model.roi_heads.box_predictor.nms_thresh = 0.2  # Higher NMS threshold for fewer boxes
        model.roi_heads.mask_predictor.mask_nms_thresh = 0.2  # Higher threshold for fewer overlapping masks
        model.roi_heads.score_thresh = 0.4  # Increase the threshold for lower-confidence masks
    else:
        backbone = resnet_fpn_backbone(backbone_name='resnet50', weights=ResNet50_Weights.IMAGENET1K_V2) # ResNet50_Weights.IMAGENET1K_V1)
        model = MaskRCNN(backbone, num_classes=2)  # 2 Classes (Background + 1 Object)

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

            
            model.backbone.body.conv1 = new_conv1  # Replace the old Conv1 Layer with the new one
            
            # Modify the transform to handle 4 channels
            model.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
            
        # adjust loss weights
        model.rpn.rpn_cls_loss_weight = 1.0
        model.rpn.rpn_bbox_loss_weight = 2.0
        model.roi_heads.mask_loss_weight = 2.0
        model.roi_heads.box_loss_weight = 1.0
        model.roi_heads.classification_loss_weight = 1.0
        
        # adjust non-maximum suppression
        model.roi_heads.nms_thresh = 0.4
        model.roi_heads.box_predictor.nms_thresh = 0.4  # Higher NMS threshold for fewer boxes
        model.roi_heads.mask_predictor.mask_nms_thresh = 0.4  # Higher threshold for fewer overlapping masks
        model.roi_heads.score_thresh = 0.5  # Increase the threshold for lower-confidence masks

        
    # load weights
    if weights_path:
        try:
            if torch.cuda.is_available():
                model.load_state_dict(state_dict=torch.load(weights_path, weights_only=True)) 
            else:
                model.load_state_dict(state_dict=torch.load(weights_path, weights_only=True, map_location=torch.device('cpu'))) 
        except RuntimeError:
            if use_4_channels:
                raise RuntimeError("It seems like you try to load a model without depth. Try to set 'USE_DEPTH' to false.")
            else:
                raise RuntimeError("It seems like you try to load a model with depth. Try to set 'USE_DEPTH' to true.")

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
    
        
    log(log_path, model_str, should_log=should_log, should_print=False)
    
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

        self.extract_width_height()

        # update augmentations -> needed for background augmentation
        if self.transform:
            self.transform.update(bg_value=self.background_value, width=self.width, height=self.height)



    def __len__(self):
        return len(self.img_names)



    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        depth_path = os.path.join(self.depth_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # Load the RGB Image, Depth Image and the Gray Mask (and make sure that the size is right)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Error during data loading: there is no '{img_path}'")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.use_depth:
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if len(depth.shape) > 2:
                _, depth, _, _ =  cv2.split(depth)

        if self.use_mask:
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)    # cv2.IMREAD_GRAYSCALE)
            if len(mask.shape) > 2:
                mask = self.rgb_mask_to_grey_mask(rgb_img=mask, verify=False)

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
            
            return image, target, img_name
        else:
            return image, img_name



    def extract_width_height(self):
        for img_name in self.img_names:
            try:
                img_path = os.path.join(self.img_dir, img_name)
                
                image = cv2.imread(img_path)
                if image is None:
                    raise FileNotFoundError(f"Error during data loading: there is no '{img_path}'")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                self.width = image.shape[1]
                self.height = image.shape[0]

                break
            except Exception:
                pass
            


    def size(self):
        try:
            return self.height, self.width
        except Exception:
            self.extract_width_height()
            return self.height, self.width


    def get_bounding_boxes(self, masks):
        boxes = []
        
        for mask in masks:
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
                images_found += 1
            else:
                images_not_found += [image_path]

            # Check Depth Image
            if self.use_depth:
                depth_path = os.path.join(self.depth_dir, cur_image)
                depth_exists = os.path.exists(depth_path) and os.path.isfile(depth_path) and any([depth_path.endswith(ending) for ending in [".png", ".jpg", ".jpeg"]])

                if depth_exists:
                    depth_found += 1
                else:
                    depth_not_found += [depth_path]

            # Check Mask Image
            if self.use_mask:
                mask_path = os.path.join(self.mask_dir, cur_image)
                mask_exists = os.path.exists(mask_path) and os.path.isfile(mask_path) and any([mask_path.endswith(ending) for ending in [".png", ".jpg", ".jpeg"]])
                
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

        # verify minimum foundings
        if len(self.img_names) <= 0:
            error_message = f"DataLoader found not enough files! \n    -> {images_found} RGB images found in '{self.img_dir}'"

            if self.use_mask:
                error_message += f"\n    -> {masks_found} mask images found in '{self.mask_dir}'"
                if masks_found <= 0 and images_found > 0:
                    error_message += "\n        => are you sure you want USE_MASK to be True?"

            if self.use_depth:
                error_message += f"\n    -> {depth_found} depth images found in '{self.depth_dir}'"
                if depth_found <= 0 and images_found > 0:
                    error_message += "\n        => are you sure you want USE_DEPTH to be True?"


            raise FileNotFoundError(error_message)


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


def collate_without_mask_fn(batch):
    images, names = zip(*batch)
    
    return torch.stack(images, 0), names



# Custom Transformations
class Random_Flip:
    def __init__(self, probability=0.05):
        self.probability = probability

    def __call__(self, images):
        rgb_img, depth_img, mask_img = images
        
        if random.random() < self.probability:
            # Horizontal Flipping
            rgb_img = T.functional.hflip(rgb_img)
            if depth_img is not None:
                depth_img = T.functional.hflip(depth_img)
            if mask_img is not None:
                mask_img = T.functional.hflip(mask_img)
        if random.random() < self.probability:
            # Vertical Flipping
            rgb_img = T.functional.vflip(rgb_img)
            if depth_img is not None:
                depth_img = T.functional.vflip(depth_img)
            if mask_img is not None:
                mask_img = T.functional.vflip(mask_img)
        return rgb_img, depth_img, mask_img



class Random_Rotation:
    def __init__(self, max_angle=30, probability=0.05):
        self.max_angle = max_angle
        self.probability = probability
    
    def __call__(self, images):
        rgb_img, depth_img, mask_img = images
        
        if random.random() < self.probability:
            angle = random.uniform(-self.max_angle, self.max_angle)
            rgb_img = T.functional.rotate(rgb_img, angle)
            if depth_img is not None:
                depth_img = T.functional.rotate(depth_img, angle, interpolation=T.InterpolationMode.NEAREST)
            if mask_img is not None:
                mask_img = T.functional.rotate(mask_img, angle, interpolation=T.InterpolationMode.NEAREST)
        return rgb_img, depth_img, mask_img



class Random_Crop:
    def __init__(self, min_crop_size, max_crop_size, probability=0.05):
        # min_crop_size und max_crop_size sind Tupel (min_h, min_w), (max_h, max_w)
        self.min_crop_size = min_crop_size
        self.max_crop_size = max_crop_size
        self.probability = probability
    
    def __call__(self, images):
        rgb_img, depth_img, mask_img = images
        
        if random.random() < self.probability:
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
    def __init__(self, brightness_range=0.2, contrast_range=0.2, probability=0.05):
        self.brightness = brightness_range
        self.contrast = contrast_range
        self.probability = probability

    def __call__(self, images):
        rgb_img, depth_img, mask_img = images
        
        if random.random() < self.probability:
            brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
            contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
            rgb_img = T.functional.adjust_brightness(rgb_img, brightness_factor)
            rgb_img = T.functional.adjust_contrast(rgb_img, contrast_factor)
        
        return rgb_img, depth_img, mask_img



class Add_Gaussian_Noise:
    def __init__(self, mean=0, std=0.01, probability=0.05):
        self.mean = mean
        self.std = std
        self.probability = probability
    
    def __call__(self, images):
        rgb_img, depth_img, mask_img = images

        # Convert Pillow image to NumPy array
        rgb_img_np = np.array(rgb_img)
        
        # Apply Gaussian noise to RGB image
        if random.random() < self.probability:
            noise_rgb = np.random.randn(*rgb_img_np.shape) * self.std + self.mean
            rgb_img_np = np.clip(rgb_img_np + noise_rgb, 0, 255).astype(np.uint8)
        
        # Convert back to Pillow image
        rgb_img = Image.fromarray(rgb_img_np)

        # Handle depth image if it's not None
        if random.random() < self.probability and depth_img is not None:
            depth_img_np = np.array(depth_img)
            noise_depth = np.random.randn(*depth_img_np.shape) * self.std + self.mean
            depth_img_np = np.clip(depth_img_np + noise_depth, 0, 255).astype(np.uint8)
            depth_img = Image.fromarray(depth_img_np)

        return rgb_img, depth_img, mask_img


class Random_Gaussian_Blur:
    def __init__(self, kernel_size=5, sigma=(0.1, 2.0), probability=0.05):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.probability = probability

    def __call__(self, images):
        rgb_img, depth_img, mask_img = images
        
        if random.random() < self.probability:
            sigma = random.uniform(*self.sigma)
            rgb_img = T.GaussianBlur(kernel_size=self.kernel_size, sigma=sigma)(rgb_img)
            if depth_img is not None:
                depth_img = T.GaussianBlur(kernel_size=self.kernel_size, sigma=sigma)(depth_img)
        
        return rgb_img, depth_img, mask_img



class Random_Scale:
    def __init__(self, scale_range=(0.8, 1.2), probability=0.05):
        self.scale_range = scale_range
        self.probability = probability
    
    def __call__(self, images):
        rgb_img, depth_img, mask_img = images
        
        if random.random() < self.probability:
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
    def __init__(self, bg_value=1, probability=0.2):
        self.bg_value = bg_value
        self.probability = probability
    
    def __call__(self, images):
        rgb_img, depth_img, mask_img = images
        rgb_img, depth_img, mask_img = pil_to_cv2([rgb_img, depth_img, mask_img])

        width = rgb_img.shape[1]
        height = rgb_img.shape[0]
        
        if random.random() < self.probability:
            mode = random.choice(["noise", "checkerboard", "gradient pattern", "color shift"])

            if mode == "noise":
                background_pattern = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            elif mode == "checkerboard":
                checker_size = random.choice([5, 10, 25, 50])
                color1 = random.randint(180, 255)
                color1 = [color1, color1, color1]    # Brighter Color
                color2 = random.randint(0, 130)
                color2 = [color2, color2, color2]    # Darker Color

                # Create the checkerboard pattern
                background_pattern = np.zeros((height, width, 3), dtype=np.uint8)
                for i in range(0, height, checker_size):
                    for j in range(0, width, checker_size):
                        color = color1 if (i // checker_size + j // checker_size) % 2 == 0 else color2
                        background_pattern[i:i+checker_size, j:j+checker_size] = color
            elif mode == "gradient pattern":
                background_pattern = np.zeros((height, width, 3), dtype=np.uint8)

                # Generate a gradient
                if random.random() > 0.5:
                    for i in range(height):
                        color_value = int(255 * (i / height))
                        background_pattern[i, :] = [color_value, color_value, color_value]
                else:
                    for i in range(width):
                        color_value = int(255 * (i / width))
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
    def __init__(self, 
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
            # added later through the update function -> need the bg value
            info_str += "Random Background Modification, "
            
            
        if len(transformations) <= 0:
            log(log_path, "Using no Data Augmentation!", should_log=should_log, should_print=should_print)
        else:
            log(log_path, info_str[:-2], should_log=should_log, should_print=should_print)
    
    
        self.transformations = transformations
        self.augmentations = T.Compose(transformations)
        self.apply_random_background_modification = apply_random_background_modification

    def update(self, bg_value=None, width=1920, height=1080):
        if self.apply_random_background_modification and bg_value is None:
            raise ValueError("If choosing apply_random_background_modification = True,you have to pass a background value to the augmentation update.")

        if self.apply_random_background_modification:
            self.transformations += [Random_Background_Modification(bg_value=bg_value)]
        
        self.transformations += [Resize(width=width, height=height)]
    
        self.augmentations = T.Compose(self.transformations)

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



def train_loop(log_path, learning_rate, momentum, num_epochs, warm_up_iter, batch_size, 
               dataset, data_loader, name, experiment_tracking,
                use_depth, weights_path, should_log=True, should_save=True,
                return_objective='model', mask_score_threshold=0.9,
                calc_metrics=False, extended_version=False):
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
    
    # get data infos
    height, width = dataset.size()
    data_size = len(dataset)
    iteration = 0
    max_iterations = int( (data_size/batch_size)*num_epochs )
    warm_up_iterations = int( (data_size/batch_size)*1 )
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = load_maskrcnn(
                weights_path=weights_path, use_4_channels=use_depth, pretrained=False, 
                log_path=log_path, should_log=should_log, should_print=should_log,
                min_size=height, max_size=width, extended_version=extended_version
            )
    model = model.to(device)

    # Optimizer
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum, nesterov=True)  #, weight_decay=decay)
    
    if extended_version:
        # Define warm-up scheduler
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warm_up_iterations)

        # Define slow decrease scheduler (CosineAnnealingLR for smooth decays)
        main_scheduler = CosineAnnealingLR(optimizer, T_max=max_iterations, eta_min=1e-5)

        # Combine the schedulers
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warm_up_iterations])
    else:
        def warm_up_and_cool_down_lr(steps):
            # warm up phase -> increase learn rate
            if steps < warm_up_iter:
                return steps / warm_up_iter
            else:
                # cool down phase -> reduce learn rate
                return 0.1 ** ((steps - warm_up_iter) // (num_epochs*(data_size//batch_size) - warm_up_iter))

        scheduler = LambdaLR(optimizer, lr_lambda=warm_up_and_cool_down_lr)

    # Experiment Tracking
    if experiment_tracking:
        # Add model for tracking
        mlflow.pytorch.log_model(model, "Mask R-CNN")

        # Init TensorBoard writer
        writer = SummaryWriter()

    # Init
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
                    
                    if experiment_tracking:
                        mlflow.log_metric(f"learnrate_{parameter_name}", param_group['lr'], step=iteration)
                        writer.add_scalar(f"learnrate_{parameter_name}", param_group['lr'], iteration)
                else:
                    current_learnrate = optimizer.param_groups[0]['lr']
                    learnrate_str += f" {current_learnrate:.0e}"
                    if experiment_tracking:
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

            # Save Model
            if should_save and epoch % 5 == 0:
                torch.save(model.state_dict(), f'./weights/{name}_epoch_{epoch:03}.pth')
    except KeyboardInterrupt:
        if should_log:
            log(log_path, "\nStopping early. Saving network...")
        if should_save:
            torch.save(model.state_dict(), f'./weights/{name}_interrupt_{epoch}.pth')

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
            
    if should_save:
        save_path_model = f'./weights/{name}_epoch_{num_epochs:03}.pth'
        torch.save(model.state_dict(), save_path_model)

    log(log_path, f"\nCongratulations!!!! Your Model trained succefull!", should_log=should_log, should_print=should_log)
    
    if should_save:
        log(log_path, f"\n    -> Your model waits here for you: '{save_path_model}'", should_log=should_log, should_print=should_log)

    if return_objective.lower() == "loss":
        return cur_total_loss
    elif return_objective.lower() == "model":
        return model
    else:
        return



def train(
        name='mask_rcnn',
        extended_version=False,
        weights_path=None,
        num_epochs=20,
        learning_rate=0.005,
        momentum=0.9,
        warm_up_iter=100,
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
        experiment_name='3xM Instance Segmentation',
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
    augmentation = Train_Augmentations(apply_random_flip=apply_random_flip, 
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
                                should_log=True, should_print=True, should_verify=verify_data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)

    # Experiment Tracking
    if using_experiment_tracking:

        if create_new_experiment:
            try:
                EXPERIMENT_ID = mlflow.create_experiment(experiment_name)
                log(log_path, f"Created Experiment '{experiment_name}' ID: {EXPERIMENT_ID}")
            except mlflow.exceptions.MlflowException:
                log(log_path, "WARNING: Please set 'CREATE_NEW_EXPERIMENT' to False!")

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
        log(log_path, f"Loaded Experiment-System: {experiment_name}")

        mlflow.set_experiment(experiment_name)

        if using_experiment_tracking:
            with mlflow.start_run():
                mlflow.set_tag("mlflow.runName", NAME)

                mlflow.log_param("name", NAME)
                mlflow.log_param("epochs", num_epochs)
                mlflow.log_param("batch_size", batch_size)
                mlflow.log_param("learnrate", learning_rate)
                mlflow.log_param("momentum", momentum)
                mlflow.log_param("warm_up_iter", warm_up_iter)

                mlflow.log_param("images_path", img_dir)
                mlflow.log_param("masks_path", mask_dir)
                mlflow.log_param("depth_path", depth_dir)

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

                train_loop(log_path=log_path, learning_rate=learning_rate, momentum=momentum, # decay=decay,
                            warm_up_iter=warm_up_iter, 
                            num_epochs=num_epochs, batch_size=batch_size, dataset=dataset, data_loader=data_loader, 
                            name=name, experiment_tracking=using_experiment_tracking, use_depth=use_depth,
                            weights_path=weights_path, should_log=True, should_save=True,
                            return_objective="None", extended_version=extended_version)

                # close experiment tracking
                if is_mlflow_active():
                    mlflow.end_run()
    else:
        train_loop(log_path=log_path, learning_rate=learning_rate, momentum=momentum, # decay=decay, 
                        warm_up_iter=warm_up_iter,
                        num_epochs=num_epochs, batch_size=batch_size, dataset=dataset, data_loader=data_loader, 
                        name=name, experiment_tracking=False, use_depth=use_depth,
                        weights_path=weights_path, should_log=True, should_save=True,
                        return_objective="None", extended_version=extended_version)






#############
# inference #
#############



DNN_INSIGHTS = {}

def hook_func(module, input, output, name):
    global DNN_INSIGHTS
    # make sure the dict exist
    try:
        DNN_INSIGHTS
    except NameError:
        DNN_INSIGHTS = {}

    try:
        DNN_INSIGHTS[name] = {
            'output': output.detach().cpu()
        }
    except AttributeError as e:
        print(f"Error: {e} Data: {input}")

def register_maskrcnn_hooks(model):
    # Hook for ResNet layer 1, first convolution
    model.backbone.body.layer1[0].conv1.register_forward_hook(
        lambda m, i, o: hook_func(m, i, o, 'resnet_layer1_conv1')
    )

    # Hook for ResNet layer 1, second convolution
    model.backbone.body.layer1[0].conv2.register_forward_hook(
        lambda m, i, o: hook_func(m, i, o, 'resnet_layer1_conv2')
    )

    # Hook for ResNet layer 1, third convolution
    model.backbone.body.layer1[0].conv3.register_forward_hook(
        lambda m, i, o: hook_func(m, i, o, 'resnet_layer1_conv3')
    )

    # Hook for ResNet layer 2, first block's first convolution
    model.backbone.body.layer2[0].conv1.register_forward_hook(
        lambda m, i, o: hook_func(m, i, o, 'resnet_layer2_conv1')
    )

    # Hook for ResNet layer 2, first block's second convolution
    model.backbone.body.layer2[0].conv2.register_forward_hook(
        lambda m, i, o: hook_func(m, i, o, 'resnet_layer2_conv2')
    )

    # Hook for ResNet layer 2, first block's third convolution
    model.backbone.body.layer2[0].conv3.register_forward_hook(
        lambda m, i, o: hook_func(m, i, o, 'resnet_layer2_conv3')
    )

    # Hook for ResNet layer 3, first block's first convolution
    model.backbone.body.layer3[0].conv1.register_forward_hook(
        lambda m, i, o: hook_func(m, i, o, 'resnet_layer3_conv1')
    )

    # Hook for ResNet layer 3, first block's second convolution
    model.backbone.body.layer3[0].conv2.register_forward_hook(
        lambda m, i, o: hook_func(m, i, o, 'resnet_layer3_conv2')
    )

    # Hook for ResNet layer 3, first block's third convolution
    model.backbone.body.layer3[0].conv3.register_forward_hook(
        lambda m, i, o: hook_func(m, i, o, 'resnet_layer3_conv3')
    )

    # Hook for ResNet layer 4, first block's first convolution
    model.backbone.body.layer4[0].conv1.register_forward_hook(
        lambda m, i, o: hook_func(m, i, o, 'resnet_layer4_conv1')
    )

    # Hook for ResNet layer 4, first block's second convolution
    model.backbone.body.layer4[0].conv2.register_forward_hook(
        lambda m, i, o: hook_func(m, i, o, 'resnet_layer4_conv2')
    )

    # Hook for ResNet layer 4, first block's third convolution
    model.backbone.body.layer4[0].conv3.register_forward_hook(
        lambda m, i, o: hook_func(m, i, o, 'resnet_layer4_conv3')
    )

    # Hook for ResNet layer 4, second block's first convolution
    model.backbone.body.layer4[1].conv1.register_forward_hook(
        lambda m, i, o: hook_func(m, i, o, 'resnet_layer4_block2_conv1')
    )

    # Hook for ResNet layer 4, second block's second convolution
    model.backbone.body.layer4[1].conv2.register_forward_hook(
        lambda m, i, o: hook_func(m, i, o, 'resnet_layer4_block2_conv2')
    )

    # Hook for ResNet layer 4, second block's third convolution
    model.backbone.body.layer4[1].conv3.register_forward_hook(
        lambda m, i, o: hook_func(m, i, o, 'resnet_layer4_block2_conv3')
    )

    # Hook for ResNet layer 4, third block's first convolution (deepest layer)
    model.backbone.body.layer4[2].conv1.register_forward_hook(
        lambda m, i, o: hook_func(m, i, o, 'resnet_layer4_block3_conv1')
    )

    # Hook for ResNet layer 4, third block's second convolution (deepest layer)
    model.backbone.body.layer4[2].conv2.register_forward_hook(
        lambda m, i, o: hook_func(m, i, o, 'resnet_layer4_block3_conv2')
    )

    # Hook for ResNet layer 4, third block's third convolution (deepest layer)
    model.backbone.body.layer4[2].conv3.register_forward_hook(
        lambda m, i, o: hook_func(m, i, o, 'resnet_layer4_block3_conv3')
    )

    # Hook for FPN layer: the first convolution in FPN lateral connection from layer 1
    model.backbone.fpn.inner_blocks[0].register_forward_hook(
        lambda m, i, o: hook_func(m, i, o, 'fpn_lateral_layer1')
    )

    # Hook for FPN layer: the first convolution in FPN lateral connection from layer 2
    model.backbone.fpn.inner_blocks[1].register_forward_hook(
        lambda m, i, o: hook_func(m, i, o, 'fpn_lateral_layer2')
    )

    # Hook for FPN layer: the first convolution in FPN lateral connection from layer 3
    model.backbone.fpn.inner_blocks[2].register_forward_hook(
        lambda m, i, o: hook_func(m, i, o, 'fpn_lateral_layer3')
    )

    # Hook for FPN layer: the first convolution in FPN lateral connection from layer 4
    model.backbone.fpn.inner_blocks[3].register_forward_hook(
        lambda m, i, o: hook_func(m, i, o, 'fpn_lateral_layer4')
    )

    # Hook for FPN output layers after merging lateral and top-down features
    model.backbone.fpn.layer_blocks[0].register_forward_hook(
        lambda m, i, o: hook_func(m, i, o, 'fpn_output_layer1')
    )

    model.backbone.fpn.layer_blocks[1].register_forward_hook(
        lambda m, i, o: hook_func(m, i, o, 'fpn_output_layer2')
    )

    model.backbone.fpn.layer_blocks[2].register_forward_hook(
        lambda m, i, o: hook_func(m, i, o, 'fpn_output_layer3')
    )

    model.backbone.fpn.layer_blocks[3].register_forward_hook(
        lambda m, i, o: hook_func(m, i, o, 'fpn_output_layer4')
    )

    # Region Proposal Network (RPN) hooks
    # Hook for RPN head's convolutional layer for objectness score
    model.rpn.head.conv.register_forward_hook(
        lambda m, i, o: hook_func(m, i, o, 'rpn_head_conv')
    )

    # Hook for RPN head's objectness prediction layer
    model.rpn.head.cls_logits.register_forward_hook(
        lambda m, i, o: hook_func(m, i, o, 'rpn_cls_logits')
    )

    # Hook for RPN head's bounding box prediction layer
    model.rpn.head.bbox_pred.register_forward_hook(
        lambda m, i, o: hook_func(m, i, o, 'rpn_bbox_pred')
    )

    # RoI Pooling or RoI Align hooks
    # Hook for the RoI Align layer (if used in Mask R-CNN)
    model.roi_heads.box_roi_pool.register_forward_hook(
        lambda m, i, o: hook_func(m, i, o, 'roi_box_pool')
    )

    # Box Head (classification and bounding box regression heads)
    # Hook for fully connected layer in the box head's feature extractor
    model.roi_heads.box_head.fc6.register_forward_hook(
        lambda m, i, o: hook_func(m, i, o, 'box_head_fc6')
    )

    model.roi_heads.box_head.fc7.register_forward_hook(
        lambda m, i, o: hook_func(m, i, o, 'box_head_fc7')
    )

    # Hook for box classifier
    model.roi_heads.box_predictor.cls_score.register_forward_hook(
        lambda m, i, o: hook_func(m, i, o, 'box_predictor_cls_score')
    )

    # Hook for box regressor
    model.roi_heads.box_predictor.bbox_pred.register_forward_hook(
        lambda m, i, o: hook_func(m, i, o, 'box_predictor_bbox_pred')
    )

    # Mask Head (for Mask R-CNN)
    # Hook for the first convolution in the mask head if mask predictions are enabled
    if hasattr(model.roi_heads, 'mask_head'):
        model.roi_heads.mask_head[0].register_forward_hook(
            lambda m, i, o: hook_func(m, i, o, 'mask_head_conv1')
        )

    # Hook for the final layer of the mask predictor
    if hasattr(model.roi_heads, 'mask_predictor'):
        model.roi_heads.mask_predictor.mask_fcn_logits.register_forward_hook(
            lambda m, i, o: hook_func(m, i, o, 'mask_predictor_fcn_logits')
        )




def plot_feature_map(tensor, should_save, save_path, should_show, title="Feature Map"):
    if len(tensor.shape) == 4:
        num_cols = min(tensor.size()[0], 3)
        num_rows = min(tensor.size()[1], 1)

        if num_rows > num_cols:
            num_rows = num_cols

        fig, all_ax = plt.subplots(num_rows, num_cols, figsize=(15, 10))

        for cur_col in range(num_cols):
            for cur_row in range(num_rows):
                feature_map_index_1 = cur_col
                feature_map_index_2 = cur_row

                if "mask_predictor" in title:
                    title = title.replace("_logits", "")
                    feature_map_index_1 = feature_map_index_1
                    feature_map_index_2 = feature_map_index_2+1

                # Extract the feature map for the ith sample
                cur_image = tensor[feature_map_index_1, feature_map_index_2].detach().cpu().numpy()  # Select channel 0, detach from graph

                if num_cols > 1 and num_rows == 1:
                    ax = all_ax[cur_col]
                elif num_cols == 1 and num_rows == 1:
                    ax = all_ax
                elif num_cols == 1 and num_rows > 1:
                    ax = all_ax[cur_row]
                elif num_cols > 1 and num_rows > 1:
                    ax = all_ax[cur_row][cur_col]
                else:
                    raise Exception(f"Error during col: {cur_col}, row: {cur_row}")
                
                ax.imshow(cur_image, cmap='viridis')
                ax.set_title(f'{title} {cur_col+1} {cur_row+1}')
                ax.axis('off')
    elif len(tensor.shape) == 2:
        plot_image = tensor.cpu().numpy()
        plt.figure(figsize=(15, 10))
        plt.imshow(plot_image, cmap='viridis')
        plt.title(title)
        plt.axis('off')
    else:
        raise ValueException(f"Tensor with shape: {tensor.shape} can't be plottet.")


    if should_save:
        plt.savefig(os.path.join(save_path, f"{title}.jpg"))

    if should_show:
        plt.show()
    
    plt.axis('off')
    plt.clf()
    plt.close()

def visualize_insights(insights, should_save, save_path, name, should_show, max_aspect_ratio=5.0, max_cols=3, channel_limit=3, batch_limit=1):
    counter = 1
    for layer_name, data in insights.items():
        try:
            plot_feature_map(data['output'], should_save, save_path, should_show, title=f"{name}_{counter:02}_{layer_name}")
        except Exception as e:
            print(f"Error during insight visualization of: {layer_name} with error: {e} and tensor: {data['output'].size()}")
        counter += 1
        


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



def extract_and_visualize_mask(masks, image=None, ax=None, visualize=True, color_map=None, soft_join=True):
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
            h, w, c = color_image.shape
            image = (image * 255).astype(np.uint8)

            if soft_join:
                alpha = 0.1
            else:
                alpha = 0.0

            #get all background pixels
            black_pixels = np.all(color_image == [0, 0, 0], axis=-1)

            # Set background to the image
            color_image[black_pixels] = image[black_pixels]

            # Set every not background pixel to blended value using formular: α⋅image+(1−α)⋅color_image
            color_image[~black_pixels] = (
                alpha * image[~black_pixels] +
                (1 - alpha) * color_image[~black_pixels]
            ).astype(np.uint8)
                

        if ax is None:
            plt.imshow(color_image)
            # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20, 15), sharey=True)
            # fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=None)
        else:  
            ax.imshow(color_image, vmin=0, vmax=255)
            ax.set_title("Instance Segmentation Mask")
            ax.axis("off")
            ax.set_facecolor('none')

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
            mask = pred_masks[idx, 0] > 0.5  
            colored_mask = np.zeros_like(image, dtype=np.uint8)
            colored_mask[mask] = [0, 0, 255]  
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
    if pixel_acc: eval_str += f"    - Pixel Accuracy = {round(pixel_acc * 100, 2)}%\n"
    if bg_fg_acc: eval_str += f"    - Background Foreground Accuracy = {round(bg_fg_acc * 100, 2)}%\n"
    if iou: eval_str += f"    - IoU = {iou}\n"
    if precision: eval_str += f"    - Precision = {round(precision * 100, 2)}%\n        -> How many positive predicted are really positive\n        -> Only BG/FG\n"
    if recall: eval_str += f"    - Recall = {round(recall * 100, 2)}%\n        -> How many positive were found\n        -> Only BG/FG\n"
    if f1_score: eval_str += f"    - F1 Score = {round(f1_score * 100, 2)}%\n        -> Harmonic mean of Precision and Recall\n"
    if dice: eval_str += f"    - Dice Coefficient = {round(dice * 100, 2)}%\n        -> Measure of overlap between predicted and actual masks\n"
    if fpr: eval_str += f"    - False Positive Rate (FPR) = {round(fpr * 100, 2)}%\n"
    if fnr: eval_str += f"    - False Negative Rate (FNR) = {round(fnr * 100, 2)}%\n"

    if should_print:
        print(eval_str)

    if should_save:
        path = os.path.join(save_path, f"{name}_eval.txt")
        with open(path, "w") as eval_file:
            eval_file.write(eval_str)
            
    return eval_str


def eval_pred(pred, ground_truth, name="instance_segmentation", should_print=True, should_save=True, save_path="./output",
                should_pixel_acc=False, should_bg_fg_acc=False, should_iou=True,
                should_f1_score=False, should_dice=False, should_fpr=False, should_fnr=False):
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
    pixel_acc = calc_metric_with_object_matching(pred, ground_truth, calc_pixel_accuracy) if should_pixel_acc else None
    bg_fg_acc = calc_metric_with_object_matching(pred, ground_truth, calc_bg_fg_acc_accuracy) if should_bg_fg_acc else None
    iou = calc_metric_with_object_matching(pred, ground_truth, calc_intersection_over_union) if should_iou else None
    precision, recall = calc_precision_and_recall(pred, ground_truth, only_bg_and_fg=True) if should_f1_score else [None, None]
    f1_score = calc_f1_score(precision, recall) if should_f1_score else None
    dice = calc_metric_with_object_matching(pred, ground_truth, calc_dice_coefficient) if should_dice else None
    fpr = calc_metric_with_object_matching(pred, ground_truth, calc_false_positive_rate) if should_fpr else None
    fnr = calc_metric_with_object_matching(pred, ground_truth, calc_false_negative_rate) if should_fnr else None

    # plot_and_save_evaluation(pixel_acc, bg_fg_acc, iou, precision, recall, f1_score, dice, fpr, fnr, name=name, should_print=should_print, should_save=should_save, save_path=save_path)

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

    pixel_acc = statistics.mean(sum_dict["pixel accuracy"]) if len(sum_dict["pixel accuracy"]) > 0 and sum_dict["pixel accuracy"][0] is not None else None
    bg_fg_acc = statistics.mean(sum_dict["background foreground accuracy"]) if len(sum_dict["background foreground accuracy"]) > 0 and sum_dict["background foreground accuracy"][0] is not None else None
    iou = statistics.mean(sum_dict["intersection over union"]) if len(sum_dict["intersection over union"]) > 0 and sum_dict["intersection over union"][0] is not None else None
    precision = statistics.mean(sum_dict["precision"]) if len(sum_dict["precision"]) > 0 and sum_dict["precision"][0] is not None else None
    recall = statistics.mean(sum_dict["recall"]) if len(sum_dict["recall"]) > 0 and sum_dict["recall"][0] is not None else None
    f1_score = statistics.mean(sum_dict["f1 score"]) if len(sum_dict["f1 score"]) > 0 and sum_dict["f1 score"][0] is not None else None
    dice = statistics.mean(sum_dict["dice"]) if len(sum_dict["dice"]) > 0 and sum_dict["dice"][0] is not None else None
    fpr = statistics.mean(sum_dict["false positive rate"]) if len(sum_dict["false positive rate"]) > 0 and sum_dict["false positive rate"][0] is not None else None
    fnr = statistics.mean(sum_dict["false negative rate"]) if len(sum_dict["false negative rate"]) > 0 and sum_dict["false negative rate"][0] is not None else None

    # save as txt file
    # plot_and_save_evaluation(pixel_acc, bg_fg_acc, iou, precision, recall, f1_score, dice, fpr, fnr, name=name, should_print=should_print, should_save=should_save, save_path=save_path)

    # save dict as pickle file
    if should_save:
        path = os.path.join(save_path, f"{name}_eval.pkl")
        with open(path, "wb") as file:  
            pickle.dump(sum_dict, file)



def inference(  
        weights_path,
        extended_version=False,
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
        should_save_mask=True,
        output_type="jpg",
        output_dir="./output",
        should_visualize_mask=True,
        should_visualize_mask_and_image=True,
        save_visualization=True,
        save_evaluation=True,
        show_visualization=False,
        show_evaluation=False,
        mask_threshold=0.9,
        show_insights=False,
        save_insights=True,
        verify_data=True,
        reset_output=False
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

    dataset = Dual_Dir_Dataset(img_dir=img_dir, depth_dir=depth_dir, mask_dir=mask_dir, transform=None, 
                                amount=amount, start_idx=start_idx, end_idx=end_idx, image_name=image_name, 
                                data_mode=data_mode, use_mask=use_mask, use_depth=use_depth, log_path=None,
                                should_log=False, should_print=True, should_verify=verify_data)
    if use_mask:
        data_loader = DataLoader(dataset, batch_size=1, num_workers=num_workers, collate_fn=collate_fn)
    else:
        data_loader = DataLoader(dataset, batch_size=1, num_workers=num_workers, collate_fn=collate_without_mask_fn)

    all_images_size = len(data_loader)

    model_name = ".".join(weights_path.split("/")[-1].split(".")[:-1])

    if reset_output:
        try:
            shutil.rmtree(output_dir)
        except Exception:
            pass

    os.makedirs(output_dir, exist_ok=True)
    eval_sum_dict = dict()
    eval_dir = os.path.join(output_dir, "evaluations")
    if use_mask and save_evaluation:
        os.makedirs(eval_dir, exist_ok=True)
    visualization_dir = os.path.join(output_dir, "visualizations")
    if save_visualization and (should_visualize_mask or should_visualize_mask_and_image):
        os.makedirs(visualization_dir, exist_ok=True)
    
    height, width = dataset.size()

    with torch.no_grad():
        # create model
        model = load_maskrcnn(
                        weights_path=weights_path, use_4_channels=use_depth, 
                        pretrained=False, log_path=None, should_log=False, should_print=False,
                        min_size=height, max_size=width, extended_version=extended_version
                    )
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
                image = data[0][0].to(device).unsqueeze(0)
                name = data[1][0]

            # inference
            print(f"Inference with image '{name}'...")
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

            cleaned_name = model_name + "_" + ".".join(name.split(".")[:-1])

            extracted_mask = extract_and_visualize_mask(result_masks, image=None, ax=None, visualize=False, color_map=None, soft_join=False)
            if len(extracted_mask.shape) == 3 and extracted_mask.shape[2] == 1:
                extracted_mask = np.squeeze(extracted_mask, axis=2)
        
            if should_save_mask:
                print("Saving the inference segmentation mask...")
                if output_type in ["numpy", "npy"]:
                    np.save(os.path.join(output_dir, f'{cleaned_name}.npy'), extracted_mask)
                else:
                    # recommended
                    cv2.imwrite(os.path.join(output_dir, f'{cleaned_name}.png'), extracted_mask)

            # plot results
            if should_visualize_mask or should_visualize_mask_and_image:
                print("Visualize your model...")

                # plot image and mask
                if should_visualize_mask_and_image:
                    _, image_plot, color_map = extract_and_visualize_mask(result_masks, image=image, ax=None, visualize=True, color_map=None, soft_join=True)    

                    if save_visualization:
                        cv2.imwrite(os.path.join(visualization_dir, f"{cleaned_name}.jpg"), cv2.cvtColor(image_plot, cv2.COLOR_RGB2BGR))

                    if show_visualization:
                        print("\nShowing Visualization*")
                        plt.show()
                    
                    plt.clf()

                # plot mask
                if should_visualize_mask:
                    _, image_plot, color_map = extract_and_visualize_mask(result_masks, image=None, ax=None, visualize=True, color_map=color_map)    # color_map)

                    if save_visualization:
                        cv2.imwrite(os.path.join(visualization_dir, f"{cleaned_name}_mask.jpg"), cv2.cvtColor(image_plot, cv2.COLOR_RGB2BGR))

                    if show_visualization:
                        plt.show()
                    
                    plt.clf()

            # eval and plot ground truth comparisson
            if use_mask:
                if show_evaluation or save_evaluation: 
                    print("Evaluating your model...")
                if show_evaluation:
                    print("Plot result in comparison to ground truth and evaluate with ground truth*")
                # mask = cv2.resize(mask, extracted_mask.shape[1], extracted_mask.shape[0])
                masks_gt = extract_and_visualize_mask(masks_gt, image=None, ax=None, visualize=False, color_map=None, soft_join=False)
                eval_results = eval_pred(extracted_mask, masks_gt, name=cleaned_name, should_print=show_evaluation, should_save=save_evaluation, save_path=eval_dir)
                eval_sum_dict = update_evaluation_summary(sum_dict=eval_sum_dict, results=eval_results)

                if should_visualize_mask:
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
                print("Looking insight your Mask RCNN...")
                img_name = ".".join(name.split(".")[:-1])
                inisght_dir = os.path.join(visualization_dir, "dnn_insights")
                os.makedirs(inisght_dir, exist_ok=True)
                visualize_insights(insights=DNN_INSIGHTS, should_save=save_insights, save_path=inisght_dir, name=img_name, should_show=show_insights)
                          
            idx += 1
            DNN_INSIGHTS = {}
            
            plt.close()

        if use_mask:
            save_and_show_evaluation_summary(eval_sum_dict, name=model_name, should_print=show_evaluation, should_save=save_evaluation, save_path=eval_dir)






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
                img_folder_name = "rgb" # IMG_DIR.split("/")[-1]
                depth_folder_name = "depth" # DEPTH_DIR.split("/")[-1]
                mask_folder_name = "mask" # MASK_DIR.split("/")[-1]
                
                name = f"{NAME}_{cur_dataset}"
                img_path = os.path.join(MULTIPLE_DATASETS, cur_dataset, img_folder_name)
                mask_path = os.path.join(MULTIPLE_DATASETS, cur_dataset, mask_folder_name)
                depth_path = os.path.join(MULTIPLE_DATASETS, cur_dataset, depth_folder_name)
            else:
                name = NAME
                img_path = IMG_DIR
                depth_path = DEPTH_DIR
                mask_path = MASK_DIR

            train(
                name=name,
                weights_path=WEIGHTS_PATH,
                extended_version=EXTENDED_VERSION,
                num_epochs=NUM_EPOCHS,
                warm_up_iter=WARM_UP_ITER,
                learning_rate=LEARNING_RATE,
                momentum=MOMENTUM,
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
                apply_random_flip=APPLY_RANDOM_FLIP, 
                apply_random_rotation=APPLY_RANDOM_ROTATION,
                apply_random_crop=APPLY_RANDOM_CROP, 
                apply_random_brightness_contrast=APPLY_RANDOM_BRIGHTNESS_CONTRAST,
                apply_random_gaussian_noise=APPLY_RANDOM_GAUSSIAN_NOISE, 
                apply_random_gaussian_blur=APPLY_RANDOM_GAUSSIAN_BLUR,
                apply_random_scale=APPLY_RANDOM_SCALE,
                apply_random_background_modification=APPLY_RANDOM_BACKGROUND_MODIFICATION,
                verify_data=VERIFY_DATA
            )
    elif MODE == RUN_MODE.INFERENCE:
        inference(
                weights_path=WEIGHTS_PATH,
                extended_version=EXTENDED_VERSION,
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
                should_save_mask=SHOULD_SAVE_MASK,
                output_dir=OUTPUT_DIR,
                should_visualize_mask=SHOULD_VISUALIZE_MASK,
                should_visualize_mask_and_image=SHOULD_VISUALIZE_MASK_AND_IMAGE,
                save_visualization=SAVE_VISUALIZATION,
                save_evaluation=SAVE_EVALUATION,
                show_visualization=SHOW_VISUALIZATION,
                show_evaluation=SHOW_EVALUATION,
                mask_threshold=MASK_SCORE_THRESHOLD,
                show_insights=SHOW_INSIGHTS,
                save_insights=SAVE_INSIGHTS,
                verify_data=VERIFY_DATA,
                reset_output=RESET_OUTPUT
        )
    
    

