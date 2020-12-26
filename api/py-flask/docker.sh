#!/bin/bash

docker run -itd --name pyapi -p 8200:80 --mount type=bind,src=/home/hlz/stock-prediction-api/app,target=/app  hansen1416/bhs-py-api