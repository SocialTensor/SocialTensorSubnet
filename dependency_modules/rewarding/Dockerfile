FROM nvcr.io/nvidia/pytorch:23.11-py3

COPY ./ /code
RUN --mount=type=cache,target=/root/.cache pip install -r /code/requirements.txt
WORKDIR /code

EXPOSE 8000
CMD ["python", "app.py"]
