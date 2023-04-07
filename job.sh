#!/bin/bash
echo "==============JOB BEGIN============"

bash ./setup.sh

echo "===============SETUP DONE=========="

bash ./run_train.sh
echo "===============JOB END============="