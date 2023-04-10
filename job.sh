#!/bin/bash
set -x
echo "==============JOB BEGIN============"

bash ./setup.sh

echo "===============SETUP DONE=========="

#bash ./run_train.sh
#bash ./run_eval.sh
bash ./run_analysis.sh


echo "===============JOB END============="