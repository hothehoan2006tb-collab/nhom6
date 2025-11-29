# ---------- Base image ----------
FROM python:3.10-slim

# ---------- Set working directory ----------
WORKDIR /app

# ---------- Copy only dependency files first ----------
COPY requirements.txt /app/

# ---------- Install system dependencies for reportlab ----------
RUN apt-get update && apt-get install -y \
    build-essential \
    libfreetype6 \
    libfreetype6-dev \
    libjpeg62-turbo-dev \
    zlib1g-dev \
    liblcms2-dev \
    libtiff5-dev \
    libopenjp2-7-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# ---------- Install Python packages ----------
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Copy the rest of the project ----------
COPY . /app

# ---------- Expose Streamlit port ----------
EXPOSE 8501

# ---------- Streamlit config ----------
RUN mkdir -p ~/.streamlit/
RUN bash -c 'echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
port = 8501\n\
" > ~/.streamlit/config.toml'

# ---------- Command to run app ----------
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
