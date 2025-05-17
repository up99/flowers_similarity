FROM python:3.12.9

# Set working directory
WORKDIR /flower_similarity

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Expose port 5000 for Flask
EXPOSE 5000

# Run Flask app with Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app.app:app"]