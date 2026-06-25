# =====================================================
# TwentyRL Arena - Dockerfile
# =====================================================
# Multi-stage build for a small, secure production image
# =====================================================

FROM python:3.12-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FLASK_APP=app.py \
    PORT=5000

# Create non-root user for security
RUN groupadd --system --gid 1001 appuser && \
    useradd --system --uid 1001 --gid appuser --no-create-home appuser

WORKDIR /app

# Install system dependencies (curl for healthcheck)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy application code
COPY app.py game_2048.py rl_agent.py logic_engine.py math_models.py mcts.py optimization.py train.py test_suite.py index.html ./

# Copy pre-trained model & training history if they exist (optional)
COPY agent_final.pkl* ./
COPY training_history.json* ./

# Set permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail --silent http://localhost:${PORT}/ > /dev/null || exit 1

# Expose port
EXPOSE 5000

# Run with Gunicorn (production-ready WSGI server)
# - 2 workers for low-memory containers; bump for production load
# - 120s timeout to allow training requests
# - --worker-tmp-dir /tmp avoids permission issues with non-root user without home dir
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 2 --timeout 120 --worker-tmp-dir /tmp --access-logfile - app:app"]
