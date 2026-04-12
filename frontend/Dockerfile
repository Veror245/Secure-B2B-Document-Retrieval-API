# 1. Base Image
FROM python:3.13-slim

# 2. Set the working directory
WORKDIR /app

# 3. Install 'uv'
RUN pip install --no-cache-dir uv

# 4. Install dependencies globally in the container (FAST)
RUN uv pip install --system streamlit requests

# 5. Copy ONLY the frontend code (Do NOT copy pyproject.toml here)
COPY ./frontend ./frontend

# 6. Expose Streamlit's default port
EXPOSE 8501

# 7. Disable Streamlit telemetry
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_GATHER_USAGE_STATS=false

# 8. Run Streamlit directly (Do not use 'uv run')
CMD ["streamlit", "run", "frontend/app.py"]