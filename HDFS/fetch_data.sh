#!/bin/bash
# Set current working directory to script location
cd "${0%/*}" || exit

if [ -e "./data/HDFS.log" ]; then
  echo "Logs have already been downloaded and extracted!"
  exit
fi

curl https://zenodo.org/record/3227177/files/HDFS_1.tar.gz?download=1 --output ./data/hdfs.tar.gz
tar -xvzf ./data/hdfs.tar.gz --directory ./data/