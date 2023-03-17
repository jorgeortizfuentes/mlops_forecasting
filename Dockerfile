# Use an official Python runtime as a parent image
FROM python:3.8.16-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 8282 available to the world outside this container
EXPOSE 8282

# Define environment variable

# Run app.py when the container launches
CMD ["uvicorn", "main:api", "--host", "0.0.0.0", "--port", "8282"]

