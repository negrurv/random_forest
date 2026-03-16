FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY data/ /app/data/
COPY backend/ /app/backend/


WORKDIR /app/backend
RUN mkdir -p build && cd build && cmake .. && make

EXPOSE 8000


CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]