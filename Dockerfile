
FROM python:3.11-slim

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt \
    && python -m nltk.downloader stopwords

EXPOSE 8080

CMD ["streamlit", "run", "streamlit-app.py", "--server.port=8080", "--server.address=0.0.0.0"]
