import argparse
import os
import csv
def get_label_file_stats(label_file, cls_dict):
    """ convert all child classes in label files to parent classes """

    bboxes = []
    cls_ids = []
    with open(label_file, 'r') as f:
        for line in f.read().splitlines():
            cls_name, xmin, ymin, xmax, ymax = line.split(" ")
            cls_dict[cls_name] += 1

    return cls_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Class Converter')

    parser.add_argument('--data-type', type=str, default='test',
                        help='train/test/validation')

    parser.add_argument('--labels-path', type=str,
                         default=os.path.expanduser("~") + "/Data/OID/test/Person_Man_Woman_Boy_Girl_Rifle_Shotgun_Handgun_Window_Door/Label",
                         help='Directory where all labels are saved')

    parser.add_argument('--cls-names', type=str,
                        default=os.path.expanduser("~") + "/Data/OpenImagesToolKit/dataset/class.names",
                        help='file containing all cls names')

    args = parser.parse_args()
    print(args)

    with open(args.cls_names) as f:
        cls_names = f.read().splitlines()
    cls_count = [0] * len(cls_names)
    cls_dict = dict(zip(cls_names, cls_count))
    cls_count_in_dataset_dict = cls_dict.copy()
    num_images_per_cls_dict = cls_dict.copy()

    for label_file in os.listdir(args.labels_path):
        if label_file.endswith('.txt'):
            file = os.path.join(args.labels_path, label_file)
            print("counting class appearances in: {}".format(file))
            label_cls_count = get_label_file_stats(file, cls_dict.copy())
            for key in label_cls_count.keys():
                if label_cls_count[key] > 0:
                    num_images_per_cls_dict[key] += 1
                    cls_count_in_dataset_dict[key] += label_cls_count[key]

    with open(os.path.expanduser("~") + '/Data/OpenImagesToolKit/dataset/class_count_{}.csv'.format(args.data_type), 'w+') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        for key, val in cls_count_in_dataset_dict.items():
            writer.writerow([key, val])

    with open(os.path.expanduser("~") + '/Data/OpenImagesToolKit/dataset/images_per_class_{}.csv'.format(args.data_type), 'w+') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        for key, val in num_images_per_cls_dict.items():
            writer.writerow([key, val])

    print("class count in dataset: {}".format(cls_count_in_dataset_dict))
    print("images per class: {}".format(num_images_per_cls_dict))