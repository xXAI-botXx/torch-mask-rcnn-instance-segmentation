import os

import numpy as np
import cv2 

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.transforms import functional

class Dual_Dir_Dataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_names = os.listdir(img_dir)
        
        self.verify_data()

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # Load the RGB Image and the Gray Mask
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # check mask
        if len(np.unique(mask)) <= 1: 
            return None, None

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
        image = torchvision.transforms.ToTensor()(image)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        target = {
            "masks": masks,
            "boxes": self.get_bounding_boxes(masks),
            "labels": torch.ones(masks.shape[0], dtype=torch.int64)  # Set all IDs to 1 -> just one class type
        }
        
        return image, target

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
        print(f"\n{'-'*32}\nVerifying Data...")

        images_found = 0
        images_not_found = []

        masks_found = 0
        masks_not_found = []

        for cur_image in self.img_names:
            image_path = os.path.join(self.img_dir, cur_image)
            mask_path = os.path.join(self.img_dir, cur_image)

            image_exists = os.path.exists(image_path) and os.path.isfile(image_path) and any([image_path.endswith(ending) for ending in [".png", ".jpg", ".jpeg"]])
            if image_exists:
                images_found += 1
            else:
                images_not_found += [image_path]

            mask_exists = os.path.exists(mask_path) and os.path.isfile(mask_path) and any([mask_path.endswith(ending) for ending in [".png", ".jpg", ".jpeg"]])
            # check if mask has an object
            if mask_exists:
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if len(np.unique(mask_img)) <= 1:
                    mask_exists = False
            
            if mask_exists:
                masks_found += 1
            else:
                masks_not_found += [mask_path]

            if image_exists and mask_exists:
                updated_images += [cur_image]

        print(f"\n> > > Images < < <\nFound: {round((images_found/len(self.img_names))*100, 2)}% ({images_found}/{len(self.img_names)})")
    
        if len(images_not_found) > 0:
            print("\n Not Found:")
            
        for not_found in images_not_found:
            print(f"    -> {not_found}")

        print(f"\n> > > Masks < < <\nFound: {round((masks_found/len(self.img_names))*100, 2)}% ({masks_found}/{len(self.img_names)})")
            
        if len(masks_not_found) > 0:
            print("\n Not Found:")
            
        for not_found in masks_not_found:
            print(f"    -> {not_found}")

        print(f"\nUpdating Images...")
        print(f"From {len(self.img_names)} to {len(updated_images)} Images\n    -> Image amount reduced by {round(( 1-(len(updated_images)/len(self.img_names)) )*100, 2)}%")
            
        self.img_names = updated_images
        print(f"{'-'*32}\n")

def pad_masks(masks, max_num_objs):
    # Amount of masks/objects
    num_objs, height, width = masks.shape
    # Add empty masks so that every datapoint in the current batch have the same amount
    padded_masks = torch.zeros((max_num_objs, height, width), dtype=torch.uint8)
    padded_masks[:num_objs, :, :] = masks  # Add original masks
    return padded_masks

def collate_fn(batch):
    """
    Make sure that every datapoint in the current batch have the same amount of masks.
    """
    images, targets = zip(*batch)
    
    # Find the max number of masks/objects in current batch
    max_num_objs = max(target["masks"].shape[0] for target in targets)
    
    # Add padding
    for target in targets:
        target["masks"] = pad_masks(target["masks"], max_num_objs)
    
    return torch.stack(images, 0), targets

def train():
    # Hyperparameter
    num_epochs = 20
    learning_rate = 0.005
    batch_size = 2
    img_dir = '/home/local-admin/data/3xM/3xM_Dataset_1_1_TEST/rgb'
    mask_dir = '/home/local-admin/data/3xM/3xM_Dataset_1_1_TEST/mask-prep'

    # Dataset and DataLoader
    dataset = Dual_Dir_Dataset(img_dir, mask_dir)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)

    # Create Mask-RCNN with Feature Pyramid Network (FPN) as backbone
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    backbone = resnet_fpn_backbone('resnet50', pretrained=True)
    model = MaskRCNN(backbone, num_classes=2)  # 2 Classes (Background + 1 Object)
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)

    # Training
    model.train()
    for epoch in range(num_epochs):
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {losses.item():.4f}')

    # Save Model
    torch.save(model.state_dict(), './weights/mask_rcnn.pth')

#############
# inference #
#############
def load_model(model_path, num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Setze das Modell in den Evaluierungsmodus
    return model

def prepare_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileExistsError(f"Can't find {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = functional.to_tensor(image_rgb) 
    return image, image_tensor

def run_inference(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)  # Füge eine Batch-Dimension hinzu und verschiebe auf GPU/CPU
        predictions = model(image_tensor)
    return predictions

def visualize_results(image, predictions, score_threshold=0.5):
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
            colored_mask[mask] = [0, 0, 255]  # Färbe die Maske rot
            image = cv2.addWeighted(image, 1, colored_mask, 0.5, 0)
    
    return image

def single_inference(model_path, image_path, output_path, num_classes=2):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Load model
    model = load_model(model_path, num_classes)
    model.to(device)
    
    # prepare image
    image, image_tensor = prepare_image(image_path)
    
    # Make inference
    predictions = run_inference(model, image_tensor, device)
    
    # Visualize and save results
    result_image = visualize_results(image, predictions)
    cv2.imwrite(output_path, result_image)
    print(f"Save results on: {output_path}")


if __name__ == "__main__":
    # train()
    
    model_path = "./weights/mask_rcnn.pth" 
    image_path = "/home/local-admin/data/3xM/3xM_Dataset_1_1_TEST/rgb/3xM_4_1_1.png"  
    output_path = "./output/3xM_4_1_1.png"  
    
    single_inference(model_path, image_path, output_path)
    
    

