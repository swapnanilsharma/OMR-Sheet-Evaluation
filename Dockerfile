from python:3.6.9

RUN apt-get update
RUN apt-get install vim -y
RUN apt-get install libhunspell-dev -y

WORKDIR /app
COPY omrEvaluateAPI_Final.py /app/omrEvaluateAPI.py

RUN pip3 --no-cache-dir install -r requirements.txt

EXPOSE 9999

ENTRYPOINT ["python3"]
CMD ["omrEvaluateAPI_Final.py"]