#!/bin/bash

docker run -itd --name vue -p 4600:80 --mount type=bind,src=/home/hlz/blackhighsea/frontend,target=/app  hansen1416/bhs-vue