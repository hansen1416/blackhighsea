FROM tiangolo/uwsgi-nginx-flask:python3.8-alpine

RUN pip install torch torchvision Pillow

ENV UWSGI_CHEAPER 4
ENV UWSGI_PROCESSES 64