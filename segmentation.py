import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import fire
import io


def get_segmentor(model_path: str = 'model_weights.pth') -> nn.Module:
    classes = pd.read_csv('dataset/labels.csv')['label_list']
    id2label = classes.to_dict()
    id2label[0] = 'back_ground'
    label2id = {v: k for k, v in id2label.items()}

    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b5", ignore_mismatched_sizes=True,
                                                             num_labels=len(id2label), id2label=id2label,
                                                             label2id=label2id,
                                                             reshape_last_stage=True)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    return model


def get_segments(model, binary_image):
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    classes = pd.read_csv('dataset/labels.csv')['label_list']
    id2label = classes.to_dict()
    id2label[0] = 'back_ground'
    PALETTE = np.array([[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                        [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                        [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                        [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                        [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                        [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                        [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                        [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                        [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                        [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                        [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                        [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                        [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                        [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                        [255, 71, 0], [0, 235, 255], [0, 173, 255]])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    feature_extractor_inference = SegformerFeatureExtractor(do_random_crop=False, do_pad=False)

    pixel_values = feature_extractor_inference(input_image, return_tensors="pt").pixel_values.to(device)
    outputs = model(pixel_values=pixel_values)
    logits = outputs.logits
    # First, rescale logits to original image size
    upsampled_logits = nn.functional.interpolate(logits,
                                                 size=input_image.size[::-1],  # (height, width)
                                                 mode='bilinear',
                                                 align_corners=False)

    # Second, apply argmax on the class dimension
    seg = upsampled_logits.argmax(dim=1)[0]
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # height, width, 3\
    for label, color in enumerate(PALETTE):
        color_seg[seg.cpu().numpy() == label, :] = color
    # Convert to BGR
    color_seg = color_seg[..., ::-1]

    # Show image + mask
    img = np.array(input_image) * 0.6 + color_seg * 0.1
    img = img.astype(np.uint8)
    return Image.fromarray(img)


def inference(model_path: str = 'model_weights.pth', img_path: str = 'dataset/train/images/img_0002.jpeg',
              mask_path: str = 'dataset/train/masks/seg_0002.png'):
    image = Image.open(img_path)
    mask = Image.open(mask_path).convert('L')
    classes = pd.read_csv('dataset/labels.csv')['label_list']
    id2label = classes.to_dict()
    id2label[0] = 'back_ground'
    label2id = {v: k for k, v in id2label.items()}
    PALETTE = np.array([[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                        [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                        [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                        [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                        [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                        [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                        [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                        [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                        [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                        [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                        [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                        [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                        [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                        [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                        [255, 71, 0], [0, 235, 255], [0, 173, 255]])

    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b5", ignore_mismatched_sizes=True,
                                                             num_labels=len(id2label), id2label=id2label,
                                                             label2id=label2id,
                                                             reshape_last_stage=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    feature_extractor_inference = SegformerFeatureExtractor(do_random_crop=False, do_pad=False)

    pixel_values = feature_extractor_inference(image, return_tensors="pt").pixel_values.to(device)
    model.eval()
    outputs = model(pixel_values=pixel_values)
    logits = outputs.logits
    # First, rescale logits to original image size
    upsampled_logits = nn.functional.interpolate(logits,
                                                 size=image.size[::-1],  # (height, width)
                                                 mode='bilinear',
                                                 align_corners=False)

    # Second, apply argmax on the class dimension
    seg = upsampled_logits.argmax(dim=1)[0]
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # height, width, 3\
    for label, color in enumerate(PALETTE):
        color_seg[seg == label, :] = color
    # Convert to BGR
    color_seg = color_seg[..., ::-1]

    # Show image + mask
    img = np.array(image) * 0.6 + color_seg * 0.1
    img = img.astype(np.uint8)

    fig, axs = plt.subplots(1, 3, figsize=(20, 10))
    axs[0].imshow(img)
    axs[1].imshow(color_seg)
    axs[2].imshow(mask)
    plt.show()


if __name__ == '__main__':
    fire.Fire(inference)
