#!/bin/bash
set -x
echo "==============JOB BEGIN============"

bash ./setup.sh

echo "===============SETUP DONE=========="

bash ./run_train.sh

echo "===============JOB END============="