FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Railway uses PORT env variable
ENV PORT=8080

# Expose port
EXPOSE $PORT

# Run the application with shell to expand $PORT
CMD gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120

