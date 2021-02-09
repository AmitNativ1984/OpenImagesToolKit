import argparse
import os
import csv
import cv2
import numpy as np

def xyxy2x0y0wh(xmin, ymin, xmax, ymax):
    xmin = float(xmin)
    ymin = float(ymin)
    xmax = float(xmax)
    ymax = float(ymax)

    x0 = np.mean([xmin, xmax])
    y0 = np.mean([ymin, ymax])
    w = (xmax - xmin)
    h = (ymax - ymin)
    return x0, y0, w, h

def convert_label_file(label_file, cls_dict):
    """ convert all child classes in label files to parent classes """

    bboxes = []
    cls_ids = []
    img_file = label_file.replace("Label/", "").replace(".txt", ".jpg")
    img = cv2.imread(img_file)
    H, W, C = img.shape

    with open(label_file, 'r') as f:
        for line in f.read().splitlines():
            cls_name, xmin, ymin, xmax, ymax = line.split(" ")

            cls_id = float(cls_dict[cls_name])
            x0, y0, w, h = xyxy2x0y0wh(xmin, ymin, xmax, ymax)
            x0 /= W
            y0 /= H
            w /= W
            h /= H

            x0 = np.clip(x0, 0., 1.)
            y0 = np.clip(y0, 0., 1.)
            w = np.clip(w, 0., 1.)
            h = np.clip(h, 0., 1.)

            bbox = " ".join([str(int(cls_id)), str(x0), str(y0), str(w), str(h)])
            bboxes.append(bbox)

    with open(label_file, 'w') as f:
        for bbox in bboxes:
            print(bbox, file=f)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Class Converter')

    parser.add_argument('--labels-path', type=str,
                         default=os.path.expanduser("~") + "/Data/OID/test/Person_Man_Woman_Boy_Girl_Rifle_Shotgun_Handgun_Window_Door/Label",
                         help='Directory where all labels are saved')

    parser.add_argument('--cls-names', type=str,
                        default=os.path.expanduser("~") + "/Data/OpenImagesToolKit/dataset/class.names",
                        help='file containing all cls names')

    args = parser.parse_args()

    with open(args.cls_names) as f:
        cls_names = f.read().splitlines()
    cls_id = list(range(len(cls_names)))
    cls_dict = dict(zip(cls_names, cls_id))

    for label_file in os.listdir(args.labels_path):
        if label_file.endswith('.txt'):
            file = os.path.join(args.labels_path, label_file)
            print("converting bboxes in: {}".format(file))
            convert_label_file(file, cls_dict)
