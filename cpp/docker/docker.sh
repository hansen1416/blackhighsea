#!/bin/bash

docker run -itd --name gcc1 \
--mount type=bind,src=/opt,target=/opt \
--mount type=bind,src=/home/hlz/blackhighsea/torch/cartoonGan,target=/mnt u20