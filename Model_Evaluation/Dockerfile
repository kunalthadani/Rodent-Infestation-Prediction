FROM python:3.11-slim

WORKDIR /app

# Install everything in one go, incl. your constraint file
COPY requirements.txt model_requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c model_requirements.txt

# Copy the rest of your code & tests
COPY . .

# (No ENTRYPOINT/CMD here—pytest is kicked off by docker-compose)