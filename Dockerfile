FROM python:3.10

LABEL maintainer="librecht.kuijvenhoven@hu.nl"

COPY . /

COPY requirements.txt /

RUN pip3 install -r requirements.txt

EXPOSE 5000

COPY . /app
WORKDIR /app

CMD [ "python", "./similarity_server.py" ]