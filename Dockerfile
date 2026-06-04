# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Leverage Docker cache by copying and installing requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose port 5000
EXPOSE 5000

# Define environment variable for Flask
ENV FLASK_APP=flask_api.py

# Run the Flask app using Gunicorn WSGI server
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 flask_api:app
