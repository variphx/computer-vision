FROM python:3.12
WORKDIR /app

COPY . .

ARG USER_NAME="variphx"
ARG USER_EMAIL="variphx@gmail.com"

RUN apt-get update
RUN apt-get install -y git gh
RUN git config --global init.defaultBranch main
RUN git config --global user.name ${USER_NAME}
RUN git config --global user.email ${USER_EMAIL}

RUN pip install --no-cache -r requirements.txt
