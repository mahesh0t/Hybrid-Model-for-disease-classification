FROM python:3.9
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Set work directory
WORKDIR /app


# Copy all files
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Expose Streamlit port
EXPOSE 8080

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
