FROM gcr.io/embark-shared/ml/gpu:latest
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
WORKDIR /workspace
COPY pipelines pipelines
COPY src src
ENTRYPOINT ["python", "-m", "src.app"]
