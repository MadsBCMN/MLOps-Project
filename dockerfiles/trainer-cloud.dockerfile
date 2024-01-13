FROM python:3.11.7-slim-bookworm

# Install essentials
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy necessary files from your computer to the container
COPY requirements.txt requirements.txt
# COPY pyproject.toml pyproject.toml
# COPY src/ src/
# COPY data/processed/ data/processed/

# Install dependencies
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
# RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir

# Get repo
RUN git clone https://github.com/MadsBCMN/MLOps-Project.git
WORKDIR MLOps-Project/

# Get data and unpack
RUN python src/data/unpack_data.py

CMD ["python", "-u", "src/train_model_lightning.py"]