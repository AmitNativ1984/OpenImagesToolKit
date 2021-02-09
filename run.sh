#!/bin/bash
DATASET_TYPE="test validation"
classes="Person Man Woman Boy Girl Rifle Shotgun Handgun Window Door"
subfolder=Person_Man_Woman_Boy_Girl_Rifle_Shotgun_Handgun_Window_Door
limit=20

for dataset_type in $DATASET_TYPE
do
  echo "********************"
  echo "$dataset_type"
  python main.py downloader -y --limit $limit --Dataset ~/Data/OID/ --type_csv "$dataset_type" --multiclasses 1 --image_IsGroupOf 0 --classes $classes

  cp -r /home/amit/Data/OID/"$dataset_type"/$subfolder/Label /home/amit/Data/OID/"$dataset_type"/$subfolder/Label_Org

  echo "converting to class_parent"
  python modules/class_converter.py --labels-path /home/amit/Data/OID/"$dataset_type"/$subfolder/Label

  echo "IoU filtering"
  python modules/IoU_bbox_filter.py --labels-path /home/amit/Data/OID/"$dataset_type"/$subfolder/Label

  echo "Gathering data statistics"
  python modules/data_stats.py --data-type "$dataset_type" --labels-path /home/amit/Data/OID/"$dataset_type"/$subfolder/Label

  echo "Convreting to yolo format"
  cp -r /home/amit/Data/OID/"$dataset_type"/$subfolder/Label /home/amit/Data/OID/"$dataset_type"/$subfolder/Label_Merged_Cls
  python modules/convert_bbox_formats.py --labels-path /home/amit/Data/OID/"$dataset_type"/$subfolder/Label

  cp -r /home/amit/Data/OID/"$dataset_type"/$subfolder/Label /home/amit/Data/OID/"$dataset_type"/$subfolder/labels
  mkdir /home/amit/Data/OID/"$dataset_type"/$subfolder/images
  mv /home/amit/Data/OID/"$dataset_type"/$subfolder/*.jpg /home/amit/Data/OID/"$dataset_type"/$subfolder/images

  python /home/amit/Dev/KzirTracking/detector/yolov3/data/utils/create_data_txtfile.py --labels-path /home/amit/Data/OID/"$dataset_type"/$subfolder/labels --output-data-txtfile /home/amit/Data/OID/"$dataset_type"/$subfolder/data.txt

done