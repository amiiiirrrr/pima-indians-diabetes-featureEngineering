From python:latest
WORKDIR .
COPY requirements.txt ./home/
COPY src ./home/src/
COPY model ./home/model/
COPY data ./home/data/
RUN /usr/local/bin/python -m pip install --default-timeout=1200 --upgrade pip
RUN pip install --default-timeout=1200 -r ./home/requirements.txt
CMD [ "python", "./home/src/app.py" ]

