# -*- coding: utf-8 -*-
import numpy as np
import cv2


def _draw_bounding_boxes(img, bounding_boxes, scores, classes, names):

    def _draw_bounding_box(img, bounding_box, score, cls):
        
        text = '{} {:.2%}'.format(names[int(cls)], score)
        (x, y), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 1)

        if int(cls) == 0:    
            color = [255, 0, 255]
        else:
            color = [255, 255, 0]

        x1, y1, x2, y2 = bounding_box[:4]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        img = cv2.rectangle(img, (x1, y1 - y - base),
                            (x1 + x, y1),
                            color, -1)
        img = cv2.putText(img,
                          text,
                          (x1, y1),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 255, 255), 1)

        return img

    for bounding_box, score, cls in zip(bounding_boxes, scores, classes):
        img = _draw_bounding_box(img, bounding_box, score, cls)

    return img


def postprocess_img(img, img_size, bond_boxes=None):
    
    img_h, img_w = img.shape[:2]
    width, height = img_size

    img_scale = min(img_w / width, img_h / height)
    new_w, new_h = int(img_scale * width), int(img_scale * height)
    dw, dh = (img_w - new_w) // 2, (img_h - new_h) // 2

    img = img[dh:new_h + dh, dw:new_w + dw, :]
    img_resized = cv2.resize(img, (width, height))

    if bond_boxes is None:
        return img_resized, None
    else:
        bond_boxes = bond_boxes.astype(np.float32)
        bond_boxes[:, [0, 2]] = np.clip((bond_boxes[:, [0, 2]] - dw) / img_scale, 0., width)
        bond_boxes[:, [1, 3]] = np.clip((bond_boxes[:, [1, 3]] - dh) / img_scale, 0., height)

        return img_resized, bond_boxes


def preprocess_img(img, img_size, bond_boxes=None):
    
    img_w, img_h = img_size
    height, width, _ = img.shape

    img_scale = min(img_w / width, img_h / height)
    new_w, new_h = int(img_scale * width), int(img_scale * height)
    img_resized = cv2.resize(img, (new_w, new_h))

    img_paded = np.full(shape=[img_h, img_w, 3], dtype=np.float32, fill_value=127)
    dw, dh = (img_w - new_w) // 2, (img_h - new_h) // 2
    img_paded[dh:new_h + dh, dw:new_w + dw, :] = img_resized

    if bond_boxes is None:
        return img_paded

    else:
        bond_boxes = np.asarray(bond_boxes).astype(np.float32)
        bond_boxes[:, [0, 2]] = bond_boxes[:, [0, 2]] * img_scale + dw
        bond_boxes[:, [1, 3]] = bond_boxes[:, [1, 3]] * img_scale + dh

        return img_paded, bond_boxes

def decode_class_names(classes_path):
    with open(classes_path, 'r') as f:
        lines = f.readlines()
    classes = []
    for line in lines:
        line = line.strip()
        if line:
            classes.append(line)
    return classes

