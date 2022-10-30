FROM python:3.9.15
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE $PORT
CMD gunicorn -w 4 app:app