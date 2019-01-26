#!/usr/bin/env bash
# Make output directory
mkdir output

docker build -t yolo34py-api -f Dockerfile-api .

docker run --rm -it --name yolo34py-api -p 13131:13131 -v `pwd`/input:/YOLO3-4-Py/input -v `pwd`/output:/YOLO3-4-Py/output yolo34py-api
