FROM ubuntu:20.04

RUN apt-get update && apt-get install -y build-essential

COPY opencv /opt/opencv
COPY libtorch /opt/libtorch
COPY cos /opt/cos

COPY gan-generator.pt /opt/gan-generator.pt
COPY cartoonGan /opt/cartoonGan

WORKDIR /opt

EXPOSE 8888

CMD ["./cartoonGan"]