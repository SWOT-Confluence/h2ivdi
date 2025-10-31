# Stage 0 - Create from ubuntu:24.04 image
FROM ubuntu:24.04 as stage0

# Stage 1 - Instal C++ compiler
FROM stage0 as stage1
RUN apt-get update
RUN apt-get install -y g++
RUN apt-get install -y python3 python3-pip python3-venv

# Stage 2 - Install code
FROM stage1 as stage2
RUN mkdir -p /app/venv
RUN mkdir -p /app/H2iVDI
COPY ./bin /app/H2iVDI/bin
COPY ./H2iVDI /app/H2iVDI/H2iVDI
COPY ./src /app/H2iVDI/src
COPY ./pyproject.toml /app/H2iVDI/pyproject.toml
COPY ./README.md /app/H2iVDI/README.md
COPY ./requirements.txt /app/H2iVDI/requirements.txt
COPY ./setup.py /app/H2iVDI/setup.py
WORKDIR /app/H2iVDI
RUN python3 -m venv /app/venv
RUN /app/venv/bin/pip -v install .


# Stage 3 - Execute algorithm
FROM stage2 as stage3
LABEL version="2.2"
LABEL description="HiVDI v2.2 discharge algorithm."
LABEL maintainer="Kevin Larnier (kevin.larnier@hydro-matters.fr)"
ENV CONFLUENCE_US=1
ENTRYPOINT ["/app/venv/bin/python", "/app/H2iVDI/bin/h2ivdi_cli.py"]
