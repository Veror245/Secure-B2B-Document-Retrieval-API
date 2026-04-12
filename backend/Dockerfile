FROM python:3.13-slim

# Prevents Python from writing hidden .pyc files to disk
ENV PYTHONDONTWRITEBYTECODE=1
# Forces real-time logging so FastAPI logs appear immediately in your terminal
ENV PYTHONUNBUFFERED=1

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install 'uv' directly into the container
RUN pip install --no-cache-dir uv

# 4. Copy your dependency files first (for caching)
COPY pyproject.toml uv.lock ./

# 5. Install the dependencies using uv
# --system tells uv to install globally inside the container (no need for a venv in Docker)
RUN uv sync --frozen --no-dev

# 6. Copy the rest of your actual application code
COPY . .

# 7. Expose the port FastAPI will run on
EXPOSE 8000

# 8. The command to start the server
CMD ["uv", "run", "fastapi", "run", "backend/main.py", "--host", "0.0.0.0", "--port", "8000"]