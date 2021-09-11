FROM ubuntu:20.04

ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y build-essential cmake

COPY opencv /opt/opencv
COPY libtorch /opt/libtorch
COPY aliyun-oss-cpp-sdk/sdk/include /opt/aliyun-oss-cpp-sdk/sdk/include
COPY aliyun-oss-cpp-sdk/build/lib/libalibabacloud-oss-cpp-sdk.so /opt/aliyun-oss-cpp-sdk/build/lib/libalibabacloud-oss-cpp-sdk.so

COPY gan-generator.pt /opt/gan-generator.pt
COPY cartoonGan /opt/cartoonGan

WORKDIR /opt

EXPOSE 4602

CMD ["./cartoonGan"]
# CMD ["tail", "-f", "/dev/null"]