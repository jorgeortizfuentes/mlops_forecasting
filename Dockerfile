# Use an official Python runtime as a parent image
FROM python:3.8.16-slim-buster

# Set the working directory to /app
WORKDIR /awto_mle_challenge

# Copy the current directory contents into the container at /app
COPY . /awto_mle_challenge

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 8282 available to the world outside this container
EXPOSE 8282

# Run tests when the image is built (python -m unittest discover tests/)
RUN python -m unittest discover tests/

# Run main.py when the container launches (app/api/main.py)
CMD ["uvicorn", "api.main:awto_mle_challenge", "--host", "0.0.0.0", "--port", "8282"]

# 9 6.453 FileNotFoundError: [Errno 2] No such file or directory: '/app/awto_mle_challenge/data/wind_power_generation.csv'