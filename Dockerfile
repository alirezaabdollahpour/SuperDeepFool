# Use an official PyTorch image as a parent image
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install package itself
RUN pip install --no-cache-dir .

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run main.py when the container launches
CMD ["python", "main.py"]
