# Stage 0 - Create from Python3.10 image
FROM python:3.10-slim-buster as stage0

# Stage 1 - Instal C++ compiler
FROM stage0 as stage1
RUN apt-get update
RUN apt install -y g++

# Stage 2 - Install code
FROM stage1 as stage2
RUN mkdir -p /app/H2iVDI
COPY ./bin /app/H2iVDI/bin
COPY ./H2iVDI /app/H2iVDI/H2iVDI
#COPY ./sos_read /app/H2iVDI/sos_read
COPY ./src /app/H2iVDI/src
COPY ./pyproject.toml /app/H2iVDI/pyproject.toml
COPY ./README.md /app/H2iVDI/README.md
COPY ./requirements.txt /app/H2iVDI/requirements.txt
COPY ./setup.py /app/H2iVDI/setup.py
WORKDIR /app/H2iVDI
RUN pip3 install -r requirements.txt
RUN pip3 -v install .


# # Stage 3 - Execute algorithm
FROM stage2 as stage3
ENTRYPOINT ["python3", "/app/H2iVDI/bin/h2ivdi_cli.py"]