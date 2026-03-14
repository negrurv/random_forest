# 1. Base Image
FROM python:3.10-slim

# 2. Install Build Tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. Setup App Structure
WORKDIR /app

# 4. Install Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Folders to explicit Absolute Paths
# This ensures data is at /app/data and backend is at /app/backend
COPY data/ /app/data/
COPY backend/ /app/backend/

# 6. Build the C++ Engine
# We move into the backend/build folder specifically to compile
WORKDIR /app/backend
RUN mkdir -p build && cd build && cmake .. && make

# 7. Final Prep
EXPOSE 8000

# 8. Launch Command
# We run from /app/backend so api:app is found, but paths are now stable
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]