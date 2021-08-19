FROM ubuntu:20.04

# Set timezone and environment variables
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Add source files to docker image
# ADD .	/home/websocket

# Update and install dependencies
RUN apt-get -y update \
    && apt-get -y upgrade \
    && apt-get -y install git python3-pip

# Install project requirements
# RUN pip3 install asyncio websockets setuptools
RUN pip3 install numpy tornado opencv-python

EXPOSE 4601

WORKDIR /app
# CMD ["python3", "python-websockets_websocket-benchmark-server.py"]