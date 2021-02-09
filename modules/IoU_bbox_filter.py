import numpy as np
import argparse
import os
import torch
import torchvision

def get_cls_indices(class_names_file):
    with open(class_names_file) as f:
        cls_names = f.read().splitlines()

    cls_indices = list(range(len(cls_names)))

    cls_name2ind = dict(zip(cls_names, cls_indices))

    return cls_name2ind

def filter_bboxes_by_IoU(label_file, cls_name2index, iou_th):
    """ convert all child classes in label files to parent classes """

    bboxes = []
    cls_ids = []
    with open(label_file, 'r') as f:
        for line in f.read().splitlines():
            cls_name, xmin, ymin, xmax, ymax = line.split(" ")
            xmin = float(xmin)
            ymin = float(ymin)
            xmax = float(xmax)
            ymax = float(ymax)
            cls_id = float(cls_name2index[cls_name])

            bboxes.append(np.array([xmin, ymin, xmax, ymax]))
            cls_ids.append(np.array(cls_id))

    bboxes = torch.from_numpy(np.array(bboxes))
    cls_ids = torch.from_numpy(np.array(cls_ids))
    scores = torch.ones_like(cls_ids)

    keep = torchvision.ops.batched_nms(boxes=bboxes, scores=scores, idxs=cls_ids, iou_threshold=iou_th)

    filtered_bboxes = bboxes[keep].numpy()
    filtered_cls_ids = cls_ids[keep].numpy()

    keys = cls_name2index.keys()
    val = cls_name2index.values()
    cls_index2name = dict(zip(val, keys))

    with open(label_file, 'w') as f:
        for bbox, cls in zip(filtered_bboxes, filtered_cls_ids):

            line = " ".join([cls_index2name[cls]] + [str(x) for x in bbox])
            f.write("%s\n" % line)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Class Converter')

    parser.add_argument('--labels-path', type=str,
                         default="/home/amit/Data/OID/test/Person_Man_Woman_Boy_Girl_Rifle_Shotgun_Handgun/Label",
                         help='Directory where all labels are saved')

    parser.add_argument('--cls-names', type=str,
                        default="/home/amit/Data/OpenImagesToolKit/dataset/class.names",
                        help='file containing all cls names')

    parser.add_argument('--iou', type=float,
                        default=0.5,
                        help='IoU threshold')

    args = parser.parse_args()
    cls_name2ind = get_cls_indices(args.cls_names)
    for label_file in os.listdir(args.labels_path):
        if label_file.endswith('.txt'):
            file = os.path.join(args.labels_path, label_file)
            print("IoU thresholding on file: {}".format(file))
            filter_bboxes_by_IoU(file, cls_name2ind, args.iou)
