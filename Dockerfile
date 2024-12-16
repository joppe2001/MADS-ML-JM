# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:latest

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
# Assuming you have a requirements.txt file. If not, create one with your project dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install TensorBoard
RUN pip install tensorboard

# Make port 6006 available to the world outside this container
EXPOSE 6006

# Define environment variable
ENV NAME FlowerClassification

# Run your script when the container launches
CMD ["python", "flower_classification.py"]
