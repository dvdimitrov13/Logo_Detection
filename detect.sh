#!/bin/sh
# Description:
# Run a detect.py with prespecified parameters saving the output in yolo/runs/detect
# Additional arguements: 1:Source_dir, 2:Conf_treshold, 3:IoU_treshold

ENVIRONMENT=pytoch
source /home/dimitar/miniconda3/etc/profile.d/conda.sh

Modelname=exp14
# List of models to choose from:
# enter list 

WEIGHTS="${PWD}/runs/train/$Modelname/weights/last.pt"

if [ $# -ne 1 ]; then
	echo 'Provide source images to be tested'
else
	SOURCE=$1
fi

OUT_DIR='${PWD}/runs/detect'

cd ${PWD}/yolov5
echo ${PWD}

conda activate $ENVIRONMENT
echo $CONDA_DEFAULT_ENV

which python

pip list | grep packaging

python3 detect.py --weights $WEIGHTS --source $SOURCE --conf-thres 0.25 --iou-thres 0.45 --agnostic-nms --augment --project $OUT_DIR --name output_$Modelname --save-txt --save-conf

