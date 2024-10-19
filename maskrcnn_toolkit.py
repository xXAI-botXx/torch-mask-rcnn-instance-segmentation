# Instance Segmentation Toolkit with Mask-RCNN
# By Tobia Ippolito <3

###############
# definitions #
###############
class DATA_LOADING_MODE(Enum):
    ALL = "all"
    RANGE = "range"
    RANDOM = "random"
    SINGLE = "single"






#############
# variables #
#############
# Change these variables to your need

# Do you want to train? Else you make an inference
SHOULD_TRAIN = True

IMG_DIR ='/home/local-admin/data/3xM/3xM_Dataset_1_1_TEST/rgb'
DEPTH_DIR = '/home/local-admin/data/3xM/3xM_Dataset_1_1_TEST/depth-prep'
MASK_DIR = '/home/local-admin/data/3xM/3xM_Dataset_1_1_TEST/mask-prep'

WEIGHTS_PATH = "./weights/maskrcnn.pth"
USE_DEPTH = False

AMOUNT = 100     # for random mode
START_IDX = 0    # for range mode
END_IDX = 99     # for range mode
IMAGE_NAME = "3xM_0_10_10.jpg" # for single mode
DATA_MODE = DATA_LOADING_MODE.ALL

NUM_WORKERS = 4



# Only for training #
MULTIPLE_DATASETS = None    # pass the path to the folder if you want to create multiple models
NAME = 'mask_rcnn'

USING_EXPERIMENT_TRACKING = True
CREATE_NEW_EXPERIMENT = True    # set this Value to False when running once
# EXPERIMENT_ID = 12345
EXPERIMENT_NAME = "3xM Instance Segmentation"

NUM_EPOCHS = 20
LEARNING_RATE = 0.005
MOMENTUM = 0.9
DECAY = 0.0005
BATCH_SIZE = 2
SHUFFLE = True




# Only for inference #
OUTPUT_DIR = "./output"
USE_MASK = True
OUTPUT_TYPE = "png"    # numpy-array or png? -> png recommended
OUTPUT_DIR = "./output"
SHOULD_VISUALIZE = True
VISUALIZATION_DIR = "./output/visualizations"
SAVE_VISUALIZATION = True
SHOW_VISUALIZATION = True
SAVE_EVALUATION = True






###########
# imports #
###########

# basics
import os
from enum import Enum
from datetime import datetime, timedelta
import time
from IPython.display import clear_output

# image
import numpy as np
import matplotlib.pyplot as plt
import cv2 

# deep learning
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.transforms import functional


###########
# general #
###########

def load_maskrcnn(weights_path=None, use_4_channels=False, pretrained=True):
    backbone = resnet_fpn_backbone('resnet50', pretrained=pretrained)
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
            # torch.nn.init.kaiming_normal_(new_conv1.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')

        
        model.backbone.body.conv1 = new_conv1  # Replace the old Conv1 Layer with the new one
        
    if weights_path:
        model.load_state_dict(torch.load(weights_path)) 
    
    return model



class Dual_Dir_Dataset(Dataset):
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
                log_path=None
                ):

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.depth_dir = depth_dir
        self.transform = transform
        self.use_mask = use_mask
        self.use_depth = use_depth
        self.log_path = log_path
        self.img_names = self.load_datanames(image_path=img_dir, 
                                                amount=amount, 
                                                start_idx=start_idx, 
                                                end_idx=end_idx, 
                                                image_name=image_name, 
                                                data_mode=data_mode)
        
        self.verify_data()



    def __len__(self):
        return len(self.img_names)



    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        depth_path = os.path.join(self.depth_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # Load the RGB Image, Depth Image and the Gray Mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.use_depth:
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            _, depth, _, _ =  cv2.split(depth)

        if self.use_mask:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
            # # check mask
            # if len(np.unique(mask)) <= 1: 
            #     return None, None

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
        image = torchvision.transforms.ToTensor()(image)

        if self.use_mask:
            # Create List of Binary Masks
            obj_ids = np.unique(mask)
            obj_ids = obj_ids[obj_ids != 0]    # remove background
            masks = np.zeros((len(obj_ids), mask.shape[0], mask.shape[1]), dtype=np.uint8)

            for i, obj_id in enumerate(obj_ids):
                if obj_id == 0:  # Background
                    continue
                masks[i] = (mask == obj_id).astype(np.uint8)

            # Convert the RGB image and the masks to a torch tensor
            masks = torch.as_tensor(masks, dtype=torch.uint8)

        

            target = {
                "masks": masks,
                "boxes": self.get_bounding_boxes(masks),
                "labels": torch.ones(masks.shape[0], dtype=torch.int64)  # Set all IDs to 1 -> just one class type
            }
            
            return image, target, img_name
        else:
            return image, img_name



    def get_bounding_boxes(self, masks):
        boxes = []
        for mask in masks:
            pos = np.where(mask == 1)
            x_min = np.min(pos[1])
            x_max = np.max(pos[1])
            y_min = np.min(pos[0])
            y_max = np.max(pos[0])
            boxes.append([x_min, y_min, x_max, y_max])
        return torch.as_tensor(boxes, dtype=torch.float32)
    


    def verify_data(self):
        updated_images = []
        log(self.log_path, f"\n{'-'*32}\nVerifying Data...")

        images_found = 0
        images_not_found = []

        if self.use_mask:
            masks_found = 0
            masks_not_found = []

        if self.use_depth:
            depth_found = 0
            depth_not_found = []

        for cur_image in self.img_names:

            # Check RGB Image
            image_path = os.path.join(self.img_dir, cur_image)
            image_exists = os.path.exists(image_path) and os.path.isfile(image_path) and any([image_path.endswith(ending) for ending in [".png", ".jpg", ".jpeg"]])
            
            if image_exists:
                rgb_img = cv2.imread(image_path)
                rgb_shape = rgb_img.shape[:2]  # Height and Width (ignore channels)
                images_found += 1
            else:
                images_not_found += [image_path]

            # Check Depth Image
            if self.use_depth:
                depth_path = os.path.join(self.depth_dir, cur_image)
                depth_exists = os.path.exists(depth_path) and os.path.isfile(depth_path) and any([depth_path.endswith(ending) for ending in [".png", ".jpg", ".jpeg"]])

                if depth_exists:
                    if image_exists:
                        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                        depth_shape = depth_img.shape[:2]
                        if rgb_shape == depth_shape:
                            depth_found += 1
                        else:
                            depth_not_found += [depth_path]
                    else:
                        depth_found += 1
                else:
                    depth_not_found += [depth_path]

            # Check Mask Image
            if self.use_mask:
                mask_path = os.path.join(self.mask_dir, cur_image)
                mask_exists = os.path.exists(mask_path) and os.path.isfile(mask_path) and any([mask_path.endswith(ending) for ending in [".png", ".jpg", ".jpeg"]])
                # check if mask has an object
                if mask_exists:
                    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if len(np.unique(mask_img)) <= 1:
                        mask_exists = False
                    else:
                        if image_exists:
                            mask_shape = mask_img.shape[:2]

                            if rgb_shape != mask_shape:
                                mask_exists = False
                
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
        log(self.log_path, f"\n> > > Images < < <\nFound: {round((images_found/len(self.img_names))*100, 2)}% ({images_found}/{len(self.img_names)})")
    
        if len(images_not_found) > 0:
            log(self.log_path, "\n Not Found:")
            
        for not_found in images_not_found:
            log(self.log_path, f"    -> {not_found}")


        if self.use_depth:
            log(self.log_path, f"\n> > > Depth-images < < <\nFound: {round((depth_found/len(self.img_names))*100, 2)}% ({depth_found}/{len(self.img_names)})")
                
            if len(depth_not_found) > 0:
                log("\n Not Found:")
                
            for not_found in depth_not_found:
                log(self.log_path, f"    -> {not_found}")


        if self.use_mask:
            log(self.log_path, f"\n> > > Masks < < <\nFound: {round((masks_found/len(self.img_names))*100, 2)}% ({masks_found}/{len(self.img_names)})")
                
            if len(masks_not_found) > 0:
                log("\n Not Found:")
                
            for not_found in masks_not_found:
                log(self.log_path, f"    -> {not_found}")


        log(self.log_path, f"\nUpdating Images...")
        log(self.log_path, f"From {len(self.img_names)} to {len(updated_images)} Images\n    -> Image amount reduced by {round(( 1-(len(updated_images)/len(self.img_names)) )*100, 2)}%")
        
        # set new updated image ID's
        self.img_names = updated_images
        log(self.log_path, f"{'-'*32}\n")



    def load_datanames(
                self,
                path_to_images,
                amount,     # for random mode
                start_idx,  # for range mode
                end_idx,    # for range mode
                image_name, # for single mode
                data_mode=DATA_LOADING_MODE.ALL
            ):
        """
        Loads file paths from a specified directory based on the given mode.

        Parameters:
        path_to_images (str): The path to the directory containing images.
        amount (int): Number of random images to select (used in 'random' mode).
        start_idx (int): The starting index of the range of images to select (used in 'range' mode).
        end_idx (int): The ending index of the range of images to select (used in 'range' mode).
        image_name (str): The name of a single image to select (used in 'single' mode).
        data_mode (str, optional): The mode for selecting images. It can be one of the following:
            - 'all': Selects all images.
            - 'random': Selects a random set of images up to the specified amount.
            - 'range': Selects a range of images from start_idx to end_idx.
            - 'single': Selects a single image specified by image_name.
            Default is 'all'.

        Returns:
        list: A list of file-names of the selected images.

        Raises:
        ValueError: If an invalid data_mode is provided.

        Example:
        >>> load_data_paths('/path/to/images', amount=10, start_idx=0, end_idx=10, image_name='image1.jpg', data_mode='random')
        ['image2.jpg', 'image5.jpg', 'image8.jpg', ...]

        Notice: Detects all forms of files and directories and doesn't filter on them.
        """
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

        log(self.log_path, f"Image Indices:\n{data_indices}")
        log(self.log_path, f"Data Amount: {len(images)}")

        # data_amount = len(images)
        return images



def collate_fn(batch):
    """
    Make sure that every datapoint in the current batch have the same amount of masks.
    """
    images, targets, names = zip(*batch)
    
    # Find the max number of masks/objects in current batch
    max_num_objs = max(target["masks"].shape[0] for target in targets)
    
    # Add padding
    for target in targets:
        target["masks"] = pad_masks(target["masks"], max_num_objs)
    
    return torch.stack(images, 0), targets, names






############
# training #
############

def log(file_path, content, reset_logs=False, should_print=True):
    if file_path is None:
        return

    if not os.path.exists(file_path) or reset_logs:
        with open(file_path, "w") as f:
            f.write("")

    with open(file_path, "a") as f:
        f.write(content)

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
    now = datetime.now()
    output = f"Yolact Training - {now.hour:02}:{now.minute:02} {now.day:02}.{now.month:02}.{now.year:04}"

    detail_output = f"\n| epoch: {cur_epoch:>5} || iteration: {cur_iteration:>8} || duration: {duration:>8.3f} || ETA: {eta_str:>8} || total loss: {total_loss:>8.3f} || "
    detail_output += ''.join([f' {key}: {value:>8.3f} |' for key, value in losses])

    iterations_in_cur_epoch = cur_iteration - cur_epoch*(data_size // batch_size)
    cur_epoch_progress =  iterations_in_cur_epoch / (data_size // batch_size)
    cur_epoch_progress = min(int((cur_epoch_progress*100)//10), 10)
    cur_epoch_progress_ = max(10-cur_epoch_progress, 0)

    cur_total_progress = cur_iteration / max_iterations
    cur_total_progress = min(int((cur_total_progress*100)//10), 10)
    cur_total_progress_ = max(10-cur_total_progress, 0)

    percentage_output = f"\nTotal Progress: |{'#'*cur_total_progress}{' '*cur_total_progress_}|    Epoch Progress: |{'#'*cur_epoch_progress}{' '*cur_epoch_progress_}|"

    print_output = f"\n\n{'-'*32}\n{output}\n{detail_output}\n{percentage_output}\n"


    # print new output
    clear_output()

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
                use_depth, weights_path):
    # Create Mask-RCNN with Feature Pyramid Network (FPN) as backbone
    log(log_path, "Create the model and preparing for training...")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = load_maskrcnn(weights_path=weights_path, use_4_channels=use_depth, pretrained=True)
    model = model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=decay)

    # Experiment Tracking
    if experiment_tracking:
        # Add model for tracking -> doesn't work
        # if using_experiment_tracking:
        #     mlflow.pytorch.log_model(model, "model")

        # Init TensorBoard writer
        writer = SummaryWriter()

    # Init
    data_size = len(dataset)
    iteration = 0
    max_iterations = data_size*num_epochs
    last_time = time.time()
    times = []
    loss_avgs = dict()

    # Training
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

                # log loss avg
                for key, value in loss_dict.items():
                    if experiment_tracking:
                        # make experiment tracking
                        mlflow.log_metric(key, value, step=iteration)

                        # make tensorboard logging
                        writer.add_scalar(key, value, iteration)

                    if key in loss_avgs.keys():
                        loss_avgs[key] += [value]
                    else:
                        loss_avgs[key] = [value]

                if experiment_tracking:
                    total_loss = sum(loss_dict.values())

                    # make experiment tracking
                    mlflow.log_metric("total loss", total_loss, step=iteration)

                    # make tensorboard logging
                    writer.add_scalar("total loss", total_loss, iteration)

                # log time duration
                cur_time = time.time()
                duration = cur_time - last_time
                last_time = cur_time
                times += [duration]

                if iteration % 10 == 0:
                    eta_str = str(timedelta(seconds=(max_iterations-iteration) * np.avg(np.array(times)))).split('.')[0]
                        
                    total_loss = sum([np.avg(np.array(loss_avgs[k])) for k in loss_avgs.keys()])
                    loss_labels = [[key, np.avg(np.array(value))] for key, value in loss_avgs.items()]

                    # log & print info
                    update_output(
                        cur_epoch=epoch,
                        cur_iteration=iteration, 
                        max_iterations=max_iterations,
                        duration=duration,
                        data_size=data_size,
                        total_loss=total_loss,
                        losses=loss_labels,
                        batch_size=batch_size,
                        log_path=log_path
                    )

                    # reset
                    times = []
                    loss_avgs = dict()

                iteration += 1
                # torch.cuda.empty_cache()

            # Save Model
            torch.save(model.state_dict(), f'./weights/{name}.pth')
    except KeyboardInterrupt:
        log(log_path, "Stopping early. Saving network...")
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
    update_output(
        cur_epoch=num_epochs-1,
        cur_iteration=iteration-1, 
        max_iterations=max_iterations,
        duration=duration,
        data_size=data_size,
        total_loss=total_loss,
        losses=loss_labels,
        batch_size=batch_size,
        log_path=log_path
    )

    log(log_path, f"\nCongratulations!!!! Your Model trained succefull!\n\n Your model waits here for you: '{f'./weights/{name}.pth'}'")




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
        experiment_name='3xM Instance Segmentation'
    ):

    log_path = f"./logs/{name}.txt"
    log(log_path, "", reset_logs=True, print=False)

    log(log_path, f"xX Instance Segmentation with MASK-RCNN and PyTorch Xx\n                  -> {name} <-\n")

    # Dataset and DataLoader
    log(log_path, "Loading the data...")
    dataset = Dual_Dir_Dataset(img_dir=img_dir, depth_dir=depth_dir, mask_dir=mask_dir, transform=None, 
                                amount=amount, start_idx=start_idx, end_idx=end_idx, image_name=image_name, 
                                data_mode=data_mode, use_mask=True, use_depth=use_depth, log_path=log_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)

    # Experiment Tracking
    if using_experiment_tracking:
        import mlflow
        import mlflow.pytorch

        if create_new_experiment:
            EXPERIMENT_ID = mlflow.create_experiment(experiment_name)
            log(log_path, f"Created Experiment '{experiment_name}' ID: {EXPERIMENT_ID}")
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
                mlflow.log_param("learning_rate", learning_rate)
                mlflow.log_param("momentum", momentum)
                mlflow.log_param("decay", decay)

                mlflow.log_param("images_path", img_dir)
                mlflow.log_param("masks_path", mask_dir)
                mlflow.log_param("depth_path", depth_dir)
                # mlflow.log_param("img_width", img_width)
                # mlflow.log_param("img_height", img_height)

                mlflow.log_param("data_shuffle", shuffle)
                mlflow.log_param("data_mode", data_mode.value)
                mlflow.log_param("data_amount", amount)
                mlflow.log_param("start_idx", start_idx)
                mlflow.log_param("end_idx", end_idx)

                mlflow.log_param("train_data_size", len(dataset))

                mlflow.pytorch.autolog()

                train_loop(log_path=log_path, learning_rate=learning_rate, momentum=momentum, decay=decay, 
                            num_epochs=num_epochs, batch_size=batch_size, dataset=dataset, data_loader=data_loader, 
                            name=name, experiment_tracking=using_experiment_tracking, use_depth=use_depth,
                            weights_path=weights_path)

                # close experiment tracking
                if is_mlflow_active():
                    mlflow.end_run()
        else:
            train_loop(log_path=log_path, learning_rate=learning_rate, momentum=momentum, decay=decay, 
                            num_epochs=num_epochs, batch_size=batch_size, dataset=dataset, data_loader=data_loader, 
                            name=name, experiment_tracking=using_experiment_tracking, use_depth=use_depth,
                            weights_path=weights_path)




#############
# inference #
#############



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

            w, h, c = color_image.shape

            if soft_join == False:
                for cur_row_idx in range(w):
                    for cur_col_idx in range(h):
                        if color_image[cur_row_idx, cur_col_idx].sum() != 0:
                            image[cur_row_idx, cur_col_idx] = color_image[cur_row_idx, cur_col_idx]
                color_image = image
            else:
                # remove black baclground
                for cur_row_idx in range(w):
                    for cur_col_idx in range(h):
                        if color_image[cur_row_idx, cur_col_idx].sum() == 0:
                            color_image[cur_row_idx, cur_col_idx] = image[cur_row_idx, cur_col_idx]

                # Set the transparency levels (alpha and beta)
                alpha = 0.5  # transparency of 1. image
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



def calc_pixel_accuracy(mask_1, mask_2):
    """
    Calculate the pixel accuracy between two masks.

    Args:
        mask_1 (np.ndarray): The first mask.
        mask_2 (np.ndarray): The second mask.

    Returns:
        float: The pixel accuracy between the two masks.

    Raises:
        ValueError: If the shapes of the masks are different.
    """
    if mask_1.shape != mask_2.shape:
        raise ValueError(f"Can't calculate the pixel accuracy between the 2 masks because of different shapes: {mask_1.shape} and {mask_2.shape}")
    
    matching_pixels = np.sum(mask_1 == mask_2)
    all_pixels = np.prod(mask_1.shape)
    return matching_pixels / all_pixels



def calc_intersection_over_union(mask_1, mask_2):
    """
    Calculate the Intersection over Union (IoU) between two masks.

    Args:
        mask_1 (np.ndarray): The first mask.
        mask_2 (np.ndarray): The second mask.

    Returns:
        float: The IoU between the two masks.

    Raises:
        ValueError: If the shapes of the masks are different.
    """
    if mask_1.shape != mask_2.shape:
        raise ValueError(f"Can't calculate the IoU between the 2 masks because of different shapes: {mask_1.shape} and {mask_2.shape}")
    
    intersection = np.logical_and(mask_1, mask_2)
    union = np.logical_or(mask_1, mask_2)
    
    intersection_area = np.sum(intersection)
    union_area = np.sum(union)
    
    return intersection_area / union_area



def calc_precision_and_recall(mask_1, mask_2, only_bg_and_fg=False, aggregation="mean"):
    """
    Calculate the precision and recall between two masks.

    Args:
        mask_1 (np.ndarray): The first mask.
        mask_2 (np.ndarray): The second mask.
        only_bg_and_fg (bool): Whether to calculate only for background and foreground. Defaults to False.
        aggregation (str): Method to aggregate precision and recall values. Options are "sum", "mean", "median", "std", "var". Defaults to "mean".

    Returns:
        tuple: Precision and recall values.

    Raises:
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
        mask_1 (np.ndarray): The first mask.
        mask_2 (np.ndarray): The second mask.

    Returns:
        float: The Dice coefficient between the two masks.
    """
    intersection = np.logical_and(mask_1, mask_2)
    dice_score = 2 * np.sum(intersection) / (np.sum(mask_1) + np.sum(mask_2))
    
    return dice_score


def calc_f1_score(precision, recall):
    """
    Calculate the F1 Score based on precision and recall.

    Args:
        precision (float): The precision value.
        recall (float): The recall value.

    Returns:
        float: The F1 score.
    """
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def calc_false_positive_rate(mask_1, mask_2):
    """
    Calculate the False Positive Rate (FPR) between two masks.

    Args:
        mask_1 (np.ndarray): The first mask.
        mask_2 (np.ndarray): The second mask.

    Returns:
        float: The False Positive Rate.
    """
    FP = np.sum((mask_1 > 0) & (mask_2 == 0))
    TN = np.sum((mask_1 == 0) & (mask_2 == 0))
    
    return FP / (FP + TN) if FP + TN != 0 else 0


def calc_false_negative_rate(mask_1, mask_2):
    """
    Calculate the False Negative Rate (FNR) between two masks.

    Args:
        mask_1 (np.ndarray): The first mask.
        mask_2 (np.ndarray): The second mask.

    Returns:
        float: The False Negative Rate.
    """
    FN = np.sum((mask_1 == 0) & (mask_2 > 0))
    TP = np.sum((mask_1 > 0) & (mask_2 > 0))
    
    return FN / (FN + TP) if FN + TP != 0 else 0


def eval_pred(pred, ground_truth, name="instance_segmentation", should_print=True, should_save=True, save_path="./output"):
    """
    Evaluate prediction against ground truth by calculating pixel accuracy, IoU, precision, recall, F1-score, Dice coefficient, FPR, and FNR.

    Args:
        pred (np.ndarray): The predicted mask.
        ground_truth (np.ndarray): The ground truth mask.
        should_print (bool): Whether to print the evaluation results. Defaults to True.

    Returns:
        tuple: Evaluation metrics including pixel accuracy, IoU, precision, recall, F1 score, Dice coefficient, FPR, and FNR.
    """
    pixel_acc = calc_pixel_accuracy(pred, ground_truth)
    iou = calc_intersection_over_union(pred, ground_truth)
    precision, recall = calc_precision_and_recall(pred, ground_truth, only_bg_and_fg=True)
    f1_score = calc_f1_score(precision, recall)
    dice = calc_dice_coefficient(pred, ground_truth)
    fpr = calc_false_positive_rate(pred, ground_truth)
    fnr = calc_false_negative_rate(pred, ground_truth)

    eval_str = "\nEvaluation:\n"
    eval_str += f"    - Pixel Accuracy = {round(pixel_acc * 100, 2)}%\n"
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

    return pixel_acc, iou, precision, recall, f1_score, dice, fpr, fnr



def inference(  weights_path,
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
                show_visualization=True):
    print(f"xX Mask-RCNN Inference Xx")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using {device}")

    dataset = Dual_Dir_Dataset(img_dir=img_dir, depth_dir=depth_dir, mask_dir=mask_dir, transform=None, 
                                amount=amount, start_idx=start_idx, end_idx=end_idx, image_name=image_name, 
                                data_mode=data_mode, use_mask=use_mask, use_depth=use_depth, log_path=None)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=num_workers, collate_fn=collate_fn)

    model_name = ".".join(weights_path.split("/")[-1].split(".")[:-1])

    with torch.no_grad():
        # create model
        model = load_maskrcnn(weights_path=weights_path, use_4_channels=use_depth, pretrained=False)
        model.eval()
        model = model.to(device)

        for data in data_loader:
            if use_mask:
                images = data[0]
                masks = data[1]
                masks = masks.to(device)
                names = data[2]
            else:
                images = data[0]
                names = data[1]

            images = list(image.to(device) for image in images)    # when only one image and no list -> for batch: .unsqueeze(0)

            # # inference
            results = model(images)
            
            # save mask
            os.makedirs(output_dir, exist_ok=True)

            for idx, result in enumerate(results):    # does the results really just stacked?
                result = {key: value.to(torch.device("cpu")) for key, value in result.items()}
                cleaned_name = model_name + "_" + ".".join(names[idx].split(".")[:-1])

                extracted_mask = extract_and_visualize_mask(result['masks'], image=None, ax=None, visualize=False, color_map=None, soft_join=False)
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
                ax[0].imshow(images[idx])
                ax[0].set_title("Original")
                ax[0].axis("off")

                # plot mask alone
                _, color_image, color_map = extract_and_visualize_mask(masks, image=None, ax=ax[1], visualize=True)
                ax[1].set_title("Prediction Mask")
                ax[1].axis("off")

                # plot result
                _, _, _ = extract_and_visualize_mask(masks, image=images[idx], ax=ax[2], visualize=True, color_map=color_map)
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


            # eval and plot ground truth comparisson
            if use_mask:
                print("Plot result in comparison to ground truth and evaluate with ground truth*")
                mask = masks[idx].squeeze(0).numpy()
                mask = cv2.resize(mask, extracted_mask.shape[1], extracted_mask.shape[0])
                eval_pred(extracted_mask, mask, name=cleaned_name, should_print=True, should_save=save_evaluation, save_path=output_dir)

                if should_visualize:
                    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(20, 15), sharey=True)
                    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.1)
                    
                    # plot ground_truth
                    mask, _ = transform_mask(mask, one_dimensional=False)
                    ax[0].imshow(mask)
                    ax[0].set_title("Ground Truth Mask")
                    ax[0].axis("off")

                    # plot prediction mask
                    _, color_image, color_map = extract_and_visualize_mask(masks, image=None, ax=ax[1], visualize=True, color_map=color_map)
                    ax[1].set_title("Predicted Mask")
                    ax[1].axis("off")

                    if save_visualization:
                        plt.savefig(os.path.join(visualization_dir, f"{cleaned_name}_ground_truth.jpg"), dpi=fig.dpi)

                    if show_visualization:
                        print("\nShowing Ground Truth Visualization*")
                        plt.show()
                    else:
                        plt.clf()





if __name__ == "__main__":
    model_path = "./weights/mask_rcnn.pth" 
    image_path = "/home/local-admin/data/3xM/3xM_Dataset_1_1_TEST/rgb/3xM_4_1_1.png"  
    output_path = "./output/3xM_4_1_1.png" 

    if SHOULD_TRAIN:
        if MULTIPLE_DATASETS is not None:
            datasets = os.listdir(MULTIPLE_DATASETS)
        else:
            datasets = [1]

        for cur_dataset in datasets:
            if MULTIPLE_DATASETS is not None:
                name = f"{NAME}_{cur_dataset}"
                img_path = os.path.join(MULTIPLE_DATASETS, cur_dataset, "rgb")
                mask_path = os.path.join(MULTIPLE_DATASETS, cur_dataset, "mask-prep")
                depth_path = os.path.join(MULTIPLE_DATASETS, cur_dataset, "depth-prep")
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
                momentum=MOMENTUM,
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
                use_mask=USE_MASK,
                use_depth=USE_DEPTH,
                using_experiment_tracking=USING_EXPERIMENT_TRACKING,
                create_new_experiment=CREATE_NEW_EXPERIMENT,    
                # experiment_id=EXPERIMENT_ID,
                experiment_name=EXPERIMENT_NAME
            )
    else:
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
                show_visualization=SHOW_VISUALIZATION
        )
    
    

