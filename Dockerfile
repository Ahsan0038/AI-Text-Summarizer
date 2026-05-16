FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK stopwords corpus (done at build time to avoid runtime delay)
RUN python -c "import nltk; nltk.download('stopwords', quiet=True)"

# Copy application source
COPY . .

# Expose port used by Hugging Face Spaces
EXPOSE 7860

# Run with Gunicorn (production-ready WSGI server)
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "2", "--timeout", "120", "app:app"]
