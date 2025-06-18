# Use a base image with Python 3.10
FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    poppler-utils \
    libgl1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies using Poetry
RUN poetry config virtualenvs.create false \
 && poetry install --no-interaction --no-ansi

RUN chmod +x setup.sh && ./setup.sh

# Expose FastAPI default port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]