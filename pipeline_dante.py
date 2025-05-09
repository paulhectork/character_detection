import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import datasets.transforms as T
import os
import os, sys
import torch, json
import numpy as np

from main import build_model_main
from util.slconfig import SLConfig

from datasets import build_dataset
from util.visualizer import COCOVisualizer
from util import box_ops

folder_data = '/home/rbaena/Downloads/dantes/'
path_annotations = '/home/rbaena/projects/OCR/OCR_line/DINO/DANTES_annos'
list_book = os.listdir(folder_data)
dict_book = {}
for book in list_book:
    list_pages = os.listdir(folder_data + book)
    dict_book[book] = list_pages
    # sort
    dict_book[book].sort()

# IN DTLR FOLDER
model_config_path = "config/DINO/HWDB_full.py"  # change the path of the model config file
args = SLConfig.fromfile(model_config_path)
args.device = 'cuda:0'
args.CTC_training = False
args.CTC_loss_coef = 0.25
args.dataset_file = 'icdar_multi'
# HERE https://www.grosfichiers.com/ZpydrVdMz2q
model_checkpoint_path = "/home/rbaena/checkpoint/all_checkpoint/checkpoint.pth"  # change the path of the model checkpoint

args.coco_path = "/comp_robot/cv_public_dataset/COCO2017/"  # the path of coco
args.fix_size = False
# args.dataset_file = "icdar_classif_font" #icdar_" + args.dataset_file
dataset_val = build_dataset(image_set='train', args=args)

device = args.device
args.charset = dataset_val.charset
model, criterion, postprocessors = build_model_main(args)
checkpoint = torch.load(model_checkpoint_path, map_location='cpu')

import torch.nn as nn

device = args.device
args.charset = dataset_val.charset
model, criterion, postprocessors = build_model_main(args)
features_dim = model.class_embed[0].weight.data.shape[1]
new_charset_size = len(args.charset)
new_class_embed = nn.Linear(features_dim, new_charset_size, )
new_decoder_class_embed = nn.Linear(features_dim, new_charset_size, )
new_enc_out_class_embed = nn.Linear(features_dim, new_charset_size, )

if model.dec_pred_class_embed_share:
    class_embed_layerlist = [new_class_embed for i in range(model.transformer.num_decoder_layers)]

### Injective mapping between the new and old charset (random mapping)

# charset_without_accent = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
#                       't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
#                       'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4',
#                       '5', '6', '7', '8', '9', '!', '?', ]
# symbols = ['"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '@', '[',
#        '\\', ']', '^', '_', '`', '{', '|', '}', '~', ' ']
# accent_charset = ['à', 'á', 'â', 'ã', 'ä', 'å', 'ā', 'æ', 'ç', 'è', 'é', 'ê', 'ë', 'ì', 'í', 'î', 'ï', 'ð', 'ñ', 'ò',
#               'ó', 'ô', 'õ', 'ö', 'ō', 'ø', 'ù', 'ú', 'û', 'ü', 'ý', 'þ', 'ÿ', 'À', 'Á', 'Â', 'Ã', 'Ä', 'Å', 'Æ',
#               'Ç', 'È', 'É', 'Ê', 'Ë', 'Ì', 'Í', 'Î', 'Ï', 'Ð', 'Ñ', 'Ò', 'Ó', 'Ô', 'Õ', 'Ö', 'Ø', 'Ù', 'Ú', 'Û',
#               'Ü', 'Ý', 'Þ', 'Ÿ']
# weird_charset = ['«', '»', '—', "’", "°", "–", "œ"]
# old_charset = charset_without_accent + accent_charset + weird_charset + symbols
#
# not_mapped = []
# possible_mapping = list(range(len(old_charset)))
# mapping = {}
# for i, char in enumerate(args.charset):
#     if char in old_charset:
#         mapping[i] = old_charset.index(char)
#         possible_mapping.remove(mapping[i])
#     else:
#         not_mapped.append(char)
# print(len(mapping), len(not_mapped))
# while len(possible_mapping) < len(not_mapped):
#     possible_mapping.append(np.random.randint(0, len(old_charset)))
# possible_mapping = list(np.random.permutation(possible_mapping))
#
# for i, char in enumerate(args.charset):
#     if char not in old_charset:
#         mapping[i] = possible_mapping[0]
#         possible_mapping.pop(0)
#
# # check if all is mapped
# print(len(mapping), len(args.charset))
#
new_class_embed = nn.ModuleList(class_embed_layerlist)
# for j in range(model.transformer.num_decoder_layers):
#     for i in range(new_charset_size):
#         new_class_embed[j].weight.data[i, :] = model.class_embed[j].weight.data[mapping[i], :]
#         new_class_embed[j].bias.data[i] = model.class_embed[j].bias.data[mapping[i]]
#
#         new_decoder_class_embed.weight.data[i, :] = model.transformer.decoder.class_embed[j].weight.data[mapping[i], :]
#         new_decoder_class_embed.bias.data[i] = model.transformer.decoder.class_embed[j].bias.data[mapping[i]]
#
#         new_enc_out_class_embed.weight.data[i, :] = model.transformer.enc_out_class_embed.weight.data[mapping[i], :]
#         new_enc_out_class_embed.bias.data[i] = model.transformer.enc_out_class_embed.bias.data[mapping[i]]

model.class_embed = new_class_embed.to(device)
model.transformer.decoder.class_embed = new_decoder_class_embed.to(device)
model.transformer.enc_out_class_embed = new_enc_out_class_embed.to(device)
# model.transformer.enc_out_class_embed = new_enc_out_class_embed.to(device)

# if model.label_enc.weight.data.shape[0] < len(dataset_val.charset)+1:
model.label_enc = nn.Embedding(len(dataset_val.charset) + 1, features_dim).to(device)
checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()
model.to(device)
total_number_image = 0
for bb in dict_book:
    total_number_image += len(dict_book[bb])
it = 0
# dict_book lien avec des path => bb = pages
for bb in dict_book:
    for pp in dict_book[bb]:
        # it += 1
        # if it < 5977:
        #     continue
        idx = pp.split('.')[0]

        # WHOLE IMAGE
        path_im = folder_data + bb + '/' + pp

        image = Image.open(path_im).convert("RGB")

        path_anno = path_annotations + '/' + bb + '/' + idx + '.json'
        with open(path_anno) as f:
            # bounding boxes extracted from final poly
            data = json.load(f)
        list_bbox = []
        list_left_top = []
        list_labels = []
        for dd in data['annotations']:
            bbox_crop = dd['bbox']
            # crop image
            left, top, right, bottom = bbox_crop
            # widen bounding boxes
            left = left - 10
            right = right + 10
            top = top - 10
            bottom = bottom + 3
            try:
                list_left_top.append((left, top))
                crop = image.crop((left, top, right, bottom))

                # # creqte folder crop for save
                # if not os.path.exists('crops'):
                #     os.makedirs('crops')
                # crop.save('crops/'+pp.split('.')[0]+'_'+str(it)+'.jpg')

                transform = T.Compose([
                    T.RandomResize([800], max_size=1333),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

                orig_size = crop.size
                crop_image, _ = transform(crop, None)
                actual_size = crop_image.shape[2], crop_image.shape[1]
            except Exception as e:
                print(e)
                print('Error in page', pp)
                continue
            try:
                with torch.no_grad():
                    output = model.cuda()(crop_image[None].cuda())
                    polygones = output['pred_boxes']

                    postprocessors['bbox'].nms_iou_threshold = 0.2
                    output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]

                    boxes = output['boxes']
                    scores = output['scores']
                    labels = output['labels']
                    select_mask = scores > 0.1
                    boxes_xyxy = boxes.clone()
                    boxes = box_ops.box_xyxy_to_cxcywh(boxes)
                    boxes = boxes[select_mask]
                    scores = scores[select_mask]
                    labels = labels[select_mask]
                    box_label = [dataset_val.charset[i] for i in labels]
                    # TODO retrieve labels corresponding to " " => delete to not consider accents
                    box_label = [bytes(_string, "utf-8").decode("unicode_escape") for _string in box_label]
                    list_labels.append(box_label)

                ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(orig_size, actual_size))
                ratios_h, ratios_w = ratios[0], ratios[1]
                w, h = actual_size
                final_bboxes = boxes.cpu() * torch.tensor([w, h, w, h])  #
                final_bboxes[:, :2] -= final_bboxes[:, 2:] / 2
                final_bboxes *= torch.Tensor([ratios_w, ratios_h, ratios_w, ratios_h]) # Bboxes correspondant à la vraie image croppée
                list_bbox.append(final_bboxes)
            except Exception as e:
                print(e)
                print('Error in page', pp)
                list_bbox.append([])
                list_labels.append([])

        # create folder bb for
        if not os.path.exists('dantes_images'):
            os.makedirs('dantes_images')
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        bbox_image = []
        label_image = []
        for left_top, bbox, labels in zip(list_left_top, list_bbox, list_labels):
            for bbb in bbox:
                x, y, w, h = bbb
                x_shifted = x + left_top[0]
                y_shifted = y + left_top[1]
                rect = patches.Rectangle((x_shifted, y_shifted), w, h, linewidth=0.11, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                # Shift des bboxes pour avoir position relatives dans images
                bbox_image.append(torch.tensor([x_shifted, y_shifted, w, h]).long())
            label_image.append(labels)
        try:
            bbox_image = torch.stack(bbox_image)
        except Exception as e:
            print(e)
            print('Error in page', pp)
            continue

        # flatten labels
        label_image = [item for sublist in label_image for item in sublist]
        if not os.path.exists('dantes_images/' + bb):
            os.makedirs('dantes_images/' + bb)

        plt.savefig('dantes_images/' + bb + '/' + pp, dpi=300)
        if not os.path.exists('dantes_coco'):
            os.makedirs('dantes_coco')

        # convert to coco format and save json
        coco_format = {
            'images': [{'file_name': pp, 'id': 0, 'height': image.size[1], 'width': image.size[0]}],
            'annotations': [],
            'categories': []
        }

        # To keep track of category IDs
        category_map = {}

        for i, (bbox, label) in enumerate(zip(bbox_image, label_image)):
            # Add category to the categories list if it doesn't exist
            if label not in category_map:
                category_id = len(category_map) + 1  # Assign a new ID to this category
                category_map[label] = category_id
                coco_format['categories'].append({'id': category_id, 'name': label, 'supercategory': 'none'})

            # Add the annotation with the correct category ID
            coco_format['annotations'].append({
                'id': i,
                'image_id': 0,
                'bbox': [bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()],
                'category_id': category_map[label]
            })
        if not os.path.exists('dantes_coco/' + bb):
            os.makedirs('dantes_coco/' + bb)
        with open('dantes_coco/' + bb + '/' + pp.split('.')[0] + '.json', 'w') as f:
            json.dump(coco_format, f)

        print("\r Progress: {}/{}, done page {}".format(it, total_number_image, pp), end="")
