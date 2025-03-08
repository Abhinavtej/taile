# Use Python 3.9 as the base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install required Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download necessary NLTK data
RUN python -c "import nltk; nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger')"

# Copy application files
COPY . .

# Expose the application port
EXPOSE 8000

# Set the entrypoint
ENTRYPOINT ["sh", "-c", "python app.py"]