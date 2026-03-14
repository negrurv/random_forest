# 1. Start with a lightweight Linux machine that has Python pre-installed
FROM python:3.10-slim

# 2. Install the C++ compilers and CMake for Linux
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. Set the working directory inside the virtual machine
WORKDIR /app

# 4. Copy the Python requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your C++ code, Python API, and the football data
COPY backend/ ./backend/
COPY data/ ./data/

# 6. Navigate to the backend folder and compile the C++ engine FOR LINUX!
WORKDIR /app/backend
RUN mkdir -p build && cd build && cmake .. && make

# 7. Expose the port FastAPI uses
EXPOSE 8000

# 8. Start the Uvicorn server (0.0.0.0 allows the cloud to route traffic to it)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
