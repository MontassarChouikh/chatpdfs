# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install poppler-utils for pdf2image
RUN apt-get update && apt-get install -y \
    gcc \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*
    

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# Install uWSGI
RUN pip install uwsgi

# Expose the port that the Flask app runs on
EXPOSE 5001

# Run the Flask app
CMD ["flask", "run", "--host=0.0.0.0", "--port=5001"]
