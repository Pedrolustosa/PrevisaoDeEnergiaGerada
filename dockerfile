FROM python:3.9

# Instala o R e dependências
RUN apt-get update && \
    apt-get install -y r-base r-base-dev && \
    R -e "install.packages('forecast', repos='http://cran.us.r-project.org')"

# Instala dependências Python
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código da aplicação
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]