FROM python:3.11-slim

WORKDIR /app

# Copy the entire project
COPY . .

# Install dependencies from requirements.txt and additional packages
RUN pip install --no-cache-dir -r requirements.txt streamlit

# Create necessary directory structure
RUN mkdir -p artifacts/data_ingestion \
    artifacts/data_version \
    artifacts/model_version \
    artifacts/evaluation \ 
    plots

# Set Python path
ENV PYTHONPATH=/app:${PYTHONPATH}

# Expose ports for FastAPI and Streamlit
EXPOSE 8800 8501

# Create a startup script
RUN echo '#!/bin/bash\n\
# Start FastAPI in the background\n\
uvicorn main:app --host 0.0.0.0 --port 8800 & \n\
# Start Streamlit in the foreground\n\
streamlit run app.py --server.port=8501 --server.address=0.0.0.0\n\
' > /app/start.sh && chmod +x /app/start.sh

# Default command to run all services
CMD ["/app/start.sh"] 