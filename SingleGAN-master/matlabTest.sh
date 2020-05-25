#!/bin/bash
model_name=$1
roiPath=$2
save_path=$3
dataPath=$4
epoch=$5
/home/group/Matlab/Matlab2818a/bin/matlab  -nodesktop -nosplash -r "modelName='$model_name',roiPath='$roiPath',dataPath='$dataPath',resultSavePath='$save_path',epoch='$epoch';getMatFeatures;"