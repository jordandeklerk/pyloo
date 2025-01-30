FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements files
COPY requirements.txt .
COPY requirements-dev.txt .
COPY requirements-test.txt .
COPY requirements-docs.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-dev.txt \
    && pip install --no-cache-dir -r requirements-test.txt \
    && pip install --no-cache-dir -r requirements-docs.txt

# Copy package files
COPY . .

# Install package in development mode
RUN pip install -e .

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["python", "-c", "import pyloo; print(f'pyloo version {pyloo.__version__} installed successfully')"]