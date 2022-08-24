FROM python:3.10
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
WORKDIR /workspace
COPY src src
ENTRYPOINT ["python", "-m", "src.app"]
