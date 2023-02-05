FROM python:3.9-slim-buster

WORKDIR /fetch_mle

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN apt-get -y update
RUN apt-get install -y ffmpeg

COPY . .

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
