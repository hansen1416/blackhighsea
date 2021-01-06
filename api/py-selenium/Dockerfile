FROM hansen1416/selenium

WORKDIR /app

# COPY ./app /app

# RUN rm -rf /app/__pycache__

CMD ["gunicorn", "-w", "2", "--bind", "0.0.0.0", "--reload", "wsgi:app"]