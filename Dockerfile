# Use the official Python base image
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Run preprocessing script to generate tokenizer.pkl
RUN python data_preprocessing_json.py

# Specify the command to run your GAN training script
CMD ["python", "data_preprocessing.py"]
