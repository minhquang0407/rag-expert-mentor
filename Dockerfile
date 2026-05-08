FROM ubuntu:latest
LABEL authors="nguye"

ENTRYPOINT ["top", "-b"]