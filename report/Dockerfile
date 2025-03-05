FROM python:3.11-slim-buster

WORKDIR /app

# Dependencies, copied first to take advantage of caching.
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Copy the rest of the files.
COPY . /app