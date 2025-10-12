# Dockerfile for course-generator API
# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements if exists, else install manually
COPY ./requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt || true

# Copy all source code
COPY . .


EXPOSE 8000

# Start FastAPI app (asignando puerto por defecto si no está definido)
CMD ["/bin/sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${COURSE_GENERATOR_PORT:-8000}"]
