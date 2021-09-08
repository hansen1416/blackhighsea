FROM ubuntu:20.04

# Set timezone and environment variables
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# ADD .	/home/websocket

# Update and install dependencies
RUN apt-get -y update \
    && apt-get -y upgrade \
    && apt-get -y install git python3-pip \
    && apt-get install -y python3-opencv

# Install project requirements
# RUN pip3 install asyncio websockets setuptools

RUN pip3 install numpy tornado opencv-python oss2 \
    torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

EXPOSE 4601

WORKDIR /app
# Add source files to docker image
COPY ./app/httpserver.py /app

CMD ["python3", "-u", "./httpserver.py"]