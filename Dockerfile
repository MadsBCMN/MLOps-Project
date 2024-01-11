FROM python:3.9-slim
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /requirements.txt
COPY ./pyproject.toml /pyproject.toml
COPY ./src /src
COPY ./data/proces /data/proces

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]
