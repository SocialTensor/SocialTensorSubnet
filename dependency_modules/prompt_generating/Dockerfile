FROM nvcr.io/nvidia/pytorch:23.11-py3

COPY ./ /code
RUN --mount=type=cache,target=/root/.cache pip install -r /code/requirements.txt
EXPOSE 8001

WORKDIR /code

CMD ["python", "app.py"]
