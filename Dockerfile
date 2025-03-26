# Use Python 3.10 slim image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create a non-root user and set permissions
RUN useradd -m appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose the Flask app port
EXPOSE 8000

# Set environment variables
ENV SERVER_PORT=8000
ENV SERVER_HEADLESS=true
ENV SERVER_ENABLE_CORS=false

# Run the Flask app with Gunicorn
ENTRYPOINT ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "app:app"]
