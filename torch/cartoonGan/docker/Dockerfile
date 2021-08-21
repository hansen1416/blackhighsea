FROM ubuntu:20.04

RUN apt-get update && apt-get install -y build-essential

COPY libtorch /opt/libtorch
COPY opencv /opt/opencv

WORKDIR /sharedvol

CMD ["./cartoonGan"]
# CMD ["tail", "-f", "/dev/null"]