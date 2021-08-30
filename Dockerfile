FROM python:3.8.11-buster

COPY api /api
COPY cryptotradingindicator /cryptotradingindicator
COPY model/ /model/
COPY requirements.txt /requirements.txt
COPY /home/ivanfernandes/code/ivan-fernandes/gcp/crypto-indicator-IF.json /credentials.json

RUN pip install -r requirements.txt

CMD uvicorn api.fast:crypto_web --host 0.0.0.0 --port $PORT
