FROM python:3.8.11-buster

COPY api /api
COPY cryptotradingindicator /cryptotradingindicator
COPY model/ /model/
COPY requirements.txt /requirements.txt
COPY ../crypto-indicator-IF.json /credentials.json
COPY data/ /data/

RUN pip install -r requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
