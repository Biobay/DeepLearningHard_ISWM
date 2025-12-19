# Container Docker per Salad Cloud

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Installa dipendenze sistema
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copia requirements
COPY requirements.txt .

# Installa dipendenze Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia tutto il codice
COPY . .

# Crea directory necessarie
RUN mkdir -p models predictions runs checkpoints

# Rendi eseguibile lo script di startup
RUN chmod +x run_salad.sh

# Espone porta tensorboard (opzionale)
EXPOSE 6006

# Command di default
CMD ["bash", "run_salad.sh"]
