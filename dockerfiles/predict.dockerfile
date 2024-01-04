FROM python:3.9-slim

# Install essentials
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy necessary files from your computer to the container
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/proces/ data/proces/

# Install dependencies
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

# Set the entry point for prediction script
ENTRYPOINT ["python", "-u", "<project_name>/models/predict_model.py"]
