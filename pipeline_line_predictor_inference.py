import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision.ops import nms
import datasets.transforms as T
from typing import Tuple, List, Dict

from main import build_model_main
from util.slconfig import SLConfig

# -------------------------------------------------------
# load models

# Paths to the model configuration and checkpoint
model_config_path = "config/DINO/DINO_4scale.py"  # Path to model configuration file
model_checkpoint_path = "./logs/line_extraction_ICDAR_2025.pth"  # Path to model checkpoint

assert os.path.isfile(model_checkpoint_path), f"checkpoint model file not found at {model_checkpoint_path}"

# Load model configuration and build the model
args = SLConfig.fromfile(model_config_path)
args.device = 'cuda'
model, _, _ = build_model_main(args)

# Load model weights from the checkpoint
torch.serialization.add_safe_globals([argparse.Namespace])  # avoid torch `WeightsUnpickler` error
checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()

# -------------------------------------------------------
# helper functions

# Function to renormalize images
# Converts normalized tensors back to the original image scale

#TODO simplify
def renorm(img: torch.FloatTensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) -> torch.FloatTensor:
    assert img.dim() in [3, 4], "Input tensor must have 3 or 4 dimensions"
    if img.dim() == 3:
        assert img.size(0) == 3, "Expected 3 channels in input tensor"
        img_perm = img.permute(1, 2, 0)
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        img_res = img_perm * std + mean
        return img_res.permute(2, 0, 1)
    else:
        assert img.size(1) == 3, "Expected 3 channels in input tensor"
        img_perm = img.permute(0, 2, 3, 1)
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        img_res = img_perm * std + mean
        return img_res.permute(0, 3, 1, 2)

# Function to convert polygons to bounding boxes (used at the end)
def convert_poly_to_bbox(polygones):
    x0 = polygones[:, 0]
    y0 = polygones[:, 1]
    x1 = polygones[:, -4]
    y1 = polygones[:, -1]
    return torch.stack([x0, y0, x1, y1], dim=1)

# -------------------------------------------------------
# preprocess image

# Path to the input image
img_path = 'PR-SPCC-R-52859-000-00448.jpg'
image = Image.open(img_path)
orig_size = image.size

# Define image transformations. returns a function that will convert the image to a tensor
transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Apply transformations to the image. image is now a tensor
image, _ = transform(image, None)

# -------------------------------------------------------
# perform inference

# explanations:
# `.cuda()`
#   returns a copy of an object in CUDA memory
# `image[None]`
#   adds a new dimension at the specified position (here, `0`)
#   it is the same as `tensor.unsqueeze()` at position 0 :  https://docs.pytorch.org/docs/stable/generated/torch.unsqueeze.html
#   `image.shape`       => `torch.Size([3, 1319, 800])`
#   `image[None].shape` => `torch.Size([1, 3, 1319, 800])`
with torch.no_grad():
    output = model.cuda()(image[None].cuda())

# Extract polygons and apply filtering
polygones: torch.FloatTensor = output['pred_boxes']
mask: torch.BoolTensor = output['pred_logits'].sigmoid().max(-1)[0] > 0.1
actual_size: Tuple[int] = image.shape[2], image.shape[1]
ratios: Tuple[float] = tuple(float(s) / float(s_orig) for s, s_orig in zip(orig_size, actual_size))
h: int = image.shape[1]
w: int = image.shape[2]

# Scale polygons to match the current image size
# `final_poly`  = polygons corresponding to the original image (before resizing)
# `interm_poly` = polygons corresponding to the resized image
interm_poly: torch.FloatTensor = polygones[mask].cpu().detach() * torch.tensor([w, h]).repeat(10)
ratios_h: float = ratios[0]
ratios_w: float = ratios[1]
final_poly: torch.FloatTensor = interm_poly * torch.tensor([ratios_w, ratios_h]).repeat(10)
scores: torch.FloatTensor = output['pred_logits'][mask].sigmoid().max(-1)[0]

# Perform Non-Maximum Suppression (NMS)
# nms => filter the line bounding boxes to keep only the most probably ones
final_bboxes: torch.FloatTensor = convert_poly_to_bbox(final_poly)
nms_bboxes: torch.IntTensor = nms(final_bboxes.cuda(), scores.cuda(), iou_threshold=0.3)
interm_poly: torch.FloatTensor = interm_poly[nms_bboxes.cpu()]
final_poly: torch.FloatTensor = interm_poly * torch.tensor([ratios_w, ratios_h]).repeat(10)

# -------------------------------------------------------
# visualize results

# Renormalize the image for visualization
img_renorm = renorm(image)
img_renorm = img_renorm.permute(1, 2, 0)

# Visualize the image and polygons
fig, ax = plt.subplots(1)
ax.imshow(img_renorm)

# Add polygons to the visualization
for polygon in interm_poly:
    points = np.array(polygon).reshape(-1, 2)
    poly_patch = patches.Polygon(points, linewidth=0.5, edgecolor='r', facecolor='none')
    ax.add_patch(poly_patch)

out_path = f"{img_path.split('.')[0]}_out_bbox.{img_path.split('.')[1]}"
plt.savefig(fname=out_path, format="jpg")
