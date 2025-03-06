ARG PYTHON_VERSION=3.11-slim
FROM python:${PYTHON_VERSION}

# Update git, etc
RUN apt-get update && \
    apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create a working dir
WORKDIR /app

# Copy files.
COPY . /app

# Install requirements
RUN if [ -f "requirements.txt" ]; then \
        pip install --upgrade pip;\
        pip install -r requirements.txt; \
    else \
        echo "No requirements.txt found, skipping pip requirements install"; \
    fi

# Bash shell
CMD ["bash"]