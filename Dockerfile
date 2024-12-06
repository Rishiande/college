FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy all files to the working directory
COPY . .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that Streamlit uses
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
