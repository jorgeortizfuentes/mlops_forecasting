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

# Run main.py when the container launches (awto_mle_challenge/api/main.py)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8282"]
# uvicorn api.main:app --host 0.0.0.0 --port 8282