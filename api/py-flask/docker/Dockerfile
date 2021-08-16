FROM tiangolo/uwsgi-nginx-flask:python3.8-alpine

RUN python3 -m pip install flask-cors flask-socketio

ENV UWSGI_CHEAPER 4
ENV UWSGI_PROCESSES 64

