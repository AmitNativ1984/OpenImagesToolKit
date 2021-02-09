import yaml
import argparse
import os

def parse_yaml(yamlfile):
    with open(yamlfile) as file:
        parent_child_dict = yaml.full_load(file)

    return parent_child_dict

def convert_bbox_cls_child2parent(bbox, parent_child_dict):
    """
    convert [child, xmin, ymin, xmax, ymax] to:
            [parent, xmin, ymin, xmax, ymax]
    bbox: string
    """

    bbox = bbox.split(" ")
    child_cls = bbox[0]
    for key in parent_child_dict.keys():
        if child_cls in parent_child_dict[key]:
            parent_cls = key

    bbox[0] = parent_cls
    bbox = " ".join(bbox)
    return bbox

def convert_classes_in_label_file(label_file, parent2child_dict):
    """ convert all child classes in label files to parent classes """

    with open(label_file) as f:
        org_bboxes = f.read().splitlines()

    new_bboxes = []
    for bbox in org_bboxes:
        new_bboxes.append(convert_bbox_cls_child2parent(bbox, parent2child_dict))

    with open(label_file, 'w') as f:
        for bbox in new_bboxes:
            f.write("%s\n" % bbox)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Class Converter')

    parser.add_argument('--labels-path', type=str,
                         default=os.path.expanduser("~") + "/Data/OID/test/Person_Man_Woman_Boy_Girl_Rifle_Shotgun_Handgun/Label",
                         help='Directory where all labels are saved')

    parser.add_argument('--cls-dict', type=str,
                        default=os.path.expanduser("~") + "/Data/OpenImagesToolKit/dataset/classes_coverter.yaml")

    args = parser.parse_args()
    cls_dict = parse_yaml(args.cls_dict)

    for label_file in os.listdir(args.labels_path):
        if label_file.endswith('.txt'):
            file = os.path.join(args.labels_path, label_file)
            print("converting classes in: {}".format(file))
            convert_classes_in_label_file(file, cls_dict)
