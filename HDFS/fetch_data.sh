#!/bin/bash
# Set current working directory to script location
cd "${0%/*}" || exit

if [ -e "./data/hdfs.zip" ]; then
  echo "Logs have already been downloaded!"
  exit
fi

curl https://zenodo.org/record/8196385/files/HDFS_v1.zip?download --output ./data/hdfs.zip

unzip ./data/hdfs.zip -d ./data/ -x README.md
mv ./data/preprocessed/anomaly_label.csv ./data/.
rm -rf ./data/preprocessed
rm -rf ./data/hdfs.zip
