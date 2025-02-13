# Usar uma imagem Python leve
FROM python:3.9-slim

# Definir o diretório de trabalho
WORKDIR .

# Copiar os arquivos do projeto
COPY . .


# Instalar as dependências do Streamlit
RUN pip install --no-cache-dir -r requirements.txt

# Expor a porta usada pelo Streamlit
EXPOSE 8501

# Comando para iniciar o Streamlit com SSL
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

