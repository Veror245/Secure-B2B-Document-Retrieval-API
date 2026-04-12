# 1. Base Image
FROM python:3.13-slim

# 2. Set the working directory
WORKDIR /app

# 3. Install 'uv'
RUN pip install --no-cache-dir uv

# 4. Copy dependency files (Assuming context is the project root)
COPY pyproject.toml uv.lock ./

# 5. Install dependencies globally in the container
RUN uv sync --frozen --no-dev

# 6. Copy the frontend code
COPY ./frontend ./frontend

# 7. Expose Streamlit's default port
EXPOSE 8501

# 8. Disable Streamlit telemetry and run the app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_GATHER_USAGE_STATS=false

CMD ["uv", "run", "streamlit", "run", "frontend/app.py"]