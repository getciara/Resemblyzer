FROM python:3.8.0-slim-buster
MAINTAINER Ciara <info@getciara.com>

ENV FLASK_APP=ciara_diarizer\
    FLASK_ENV=production

COPY .docker/install-packages.sh .
RUN ./install-packages.sh && rm install-packages.sh

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt &&\
    apt-get purge -y build-essential

COPY . .

EXPOSE 5000

ENTRYPOINT [ "gunicorn", "ciara_diarizer:app", "--bind=0.0.0.0:5000", "--preload" ]
