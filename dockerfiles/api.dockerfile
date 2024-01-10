FROM python:3.9
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY app/ app/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
