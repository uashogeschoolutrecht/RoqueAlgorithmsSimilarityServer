FROM python:3.10

LABEL maintainer="librecht.kuijvenhoven@hu.nl"

COPY . /

COPY requirements.txt /

RUN pip install -r requirements.txt

EXPOSE 5000

COPY . /app
WORKDIR /app

CMD [ "python", "./similarity_server.py" ]