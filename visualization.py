import torch
import torch.nn.functional as F
from arch.dino_confidence import RCMNet
import numpy as np
import cv2
import os
from PIL import Image
from torchvision import transforms
import argparse

class Resize:
    def __init__(self, resize_size=532):
        self.resize_size = resize_size
    def __call__(self, image):
        resize_transform = transforms.Resize((self.resize_size, self.resize_size))
        image = resize_transform(image)
        return image

class ToTensor:
    def __call__(self, image):
        image = transforms.ToTensor()(image)
        return image

class Normalize:
    def __call__(self, image):
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        return image

def get_parser(**kwargs):
    parser = argparse.ArgumentParser(description='RCMNet')

    # Device
    parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help='Device to use for training, either "cuda:0" or "cpu".')

    # Dataset directories
    parser.add_argument('--data_root', type=str, default='./dataset', help='root directory to the dataset')
    parser.add_argument('--crop_size', type=int, default=532, help='Crop size of images.')

    # Model parameters
    parser.add_argument('--backbone_size', type=str, default='base', required=True, choices=['base', 'large', 'gaint'], help='Backbone size.')
    parser.add_argument('--decoder_type', type=str, default='segformer', help='Type of decoder to use.')

    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint', help='Checkpoint file direction.')
    parser.add_argument('--dst_dir', type=str, default='./visualization', help='visualization result save direction.')
    
    parser.add_argument('--opacity', type=float, default=0.4, help='opacity of the mask.')
    
    return parser.parse_args()

def interface(model, image: torch.Tensor, device, size):
    model.eval()
    with torch.no_grad():
        image = image.to(device).unsqueeze(0)
        pred_confidence = model.interface(image).logits

        pred_confidence = F.interpolate(pred_confidence, size=(size[1], size[0]), mode='bilinear')

        pred_confidence[pred_confidence < 0] = 0
        return pred_confidence

def process_image(model, image, image_transform, opacity=0.4):
    '''
    Function: process a single PIL image and return the original image with predictive confidence map
    Input: PIL image
    Return PIL image attached with confidence map
    '''
    transformed_image = image_transform(image)
    pred_confidence = interface(model, transformed_image, torch.device('cuda:0'), image.size)
    # attach the confidence map to the origin image
    image_con = pred_confidence.squeeze(0).cpu().permute(1, 2, 0).numpy() # (H, W, 1)
    image_con = (255 * (image_con - image_con.min()) / (image_con.max() - image_con.min())).astype(np.uint8) 
    image_con = cv2.cvtColor(image_con, cv2.COLOR_GRAY2RGB)
    image_jet = cv2.applyColorMap(image_con, cv2.COLORMAP_JET)

    mask = image_con == 0
    image_jet[mask] = 0

    image_fin = cv2.addWeighted(image_jet, opacity, np.array(image), 1, 0)

    return image_fin, image_jet


if __name__ == "__main__":
    args = get_parser()
    data_root = args.data_root
    image_root = os.path.join(data_root, 'images/validation')
    dst_dir = args.dst_dir
    
    os.makedirs(dst_dir, exist_ok=True)

    backbone_size = 'base'
    confidence_head = 'segformer'
    DEVICE = torch.device(args.device)
    image_size = args.crop_size
    confidence_normalization = False
    # checkpoint path
    checkpoint_path = args.checkpoint_path
    model = RCMNet(backbone_size=backbone_size, r=4, lora_layer=None, image_shape=(image_size,image_size), \
                        decode_type = 'linear4', confidence_head=confidence_head).to(DEVICE)
    
    model.load_parameters(checkpoint_path, DEVICE)
    image_transform = transforms.Compose([Resize(image_size), ToTensor(), Normalize()])

    for image_name in os.listdir(image_root):
        image_path = os.path.join(image_root, image_name)
        image = Image.open(image_path)
        image_fin = process_image(model, image, image_transform, args.opacity)
        cv2.imwrite(os.path.join(dst_dir, image_name), cv2.cvtColor(image_fin, cv2.COLOR_BGR2RGB))

