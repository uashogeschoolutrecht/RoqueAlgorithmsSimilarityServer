FROM python:3.10

LABEL maintainer="librecht.kuijvenhoven@hu.nl"

COPY . /

COPY requirements.txt /

RUN pip install -r requirements.txt

EXPOSE 5000

COPY . /app
WORKDIR /app

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "similarity_server:app"]