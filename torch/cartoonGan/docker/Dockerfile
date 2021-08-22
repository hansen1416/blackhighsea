FROM ubuntu:20.04

RUN apt-get update && apt-get install -y build-essential

COPY libtorch /opt/libtorch
COPY opencv /opt/opencv
COPY gan-generator.pt /opt/gan-generator.pt
COPY cartoonGan /opt/cartoonGan

WORKDIR /sharedvol

CMD ["./cartoonGan"]
# CMD ["tail", "-f", "/dev/null"]