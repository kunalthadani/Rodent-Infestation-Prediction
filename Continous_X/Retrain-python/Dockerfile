# Use an official Python runtime as a parent image
FROM python:3.11-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the requirements.txt into the container at /app
# we do this separately so that the "expensive" build step (pip install)
# does not need to be repeated if other files in /app change.
# Install system dependencies
RUN apt-get update && apt-get install -y \
    ansible \
    docker.io \
    docker-compose \
    git \
    ssh \
    && apt-get clean

# Install Ansible collection for docker_compose
RUN ansible-galaxy collection install community.docker

COPY model_requirements.txt requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -c requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Expose the port on which the app will run
EXPOSE 5000

# Run the command to start the Flask server
CMD ["python","-u","app.py"]
