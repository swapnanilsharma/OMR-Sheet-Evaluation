from python:3.6.9
MAINTAINER "swapnanil.sharmah@pwc.com"

RUN apt-get update
RUN apt-get install vim -y
RUN apt-get install libhunspell-dev -y

WORKDIR /app
COPY . /app

RUN pip3 --no-cache-dir install -r requirements.txt

EXPOSE 8999

ENTRYPOINT ["python3"]
CMD ["app.py"]