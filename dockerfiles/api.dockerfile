# api.dockerfile
FROM python:3.9

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir python-multipart
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY ./app /code/app
COPY ./models /code/models
WORKDIR /code
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
