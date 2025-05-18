# Use official Python image
FROM python:3.10.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies and only webscout[api]
# Add retry logic and split commands for better error handling
RUN apt-get update && \
    # Skip upgrades to avoid network issues - only install what's needed
    apt-get install -y --no-install-recommends gcc build-essential git && \
    pip install --upgrade pip && \
    pip install -U git+https://github.com/OEvortex/Webscout.git#egg=webscout[api] && \
    apt-get purge -y --auto-remove gcc build-essential git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the rest of the application code
COPY . /app

# Expose port for webscout API
EXPOSE 8080

# Start the webscout API server
CMD ["python", "-m", "webscout.Provider.OPENAI.api", "--port", "8080"]
