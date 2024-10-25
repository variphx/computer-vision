FROM nvidia/cuda:12.6.2-runtime-ubuntu24.04
WORKDIR /app

COPY . .

RUN apt-get update
RUN apt-get install -y python3 python3-venv python3-pip git
RUN git config --global init.defaultBranch main

ENV VIRTUAL_ENV=/app/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --no-cache-dir -r requirements.txt
