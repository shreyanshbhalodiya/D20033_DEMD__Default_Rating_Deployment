FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
WORKDIR /app/
COPY cls.pkl .
COPY requirements.txt .
COPY chrome_reviews.csv .
RUN pip install -r ./requirements.txt
COPY main.py /app/
CMD ["python", "main.py"]