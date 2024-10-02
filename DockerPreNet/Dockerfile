# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt


# Copy the current directory contents into the container at /app
COPY ../model_lib.py ../server_glob.py . /app


EXPOSE 5050

# Run the server.py script when the container launches
CMD ["python", "server_glob.py"]
