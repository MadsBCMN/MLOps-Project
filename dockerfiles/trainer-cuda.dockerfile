FROM nvcr.io/nvidia/pytorch:23.12-py3
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/proces/ data/proces/

WORKDIR /
RUN pip install -r workspace/requirements.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "workspace/src/models/train_model.py"]
