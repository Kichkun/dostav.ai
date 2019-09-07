from python:3.6

RUN apt update
RUN apt install nano
RUN apt install -y git

WORKDIR workdir

COPY workdir workdir
#RUN cp ../LKH workdir/

RUN (cd workdir/LKH-3.0.5 && make --silent)
RUN cp workdir/LKH-3.0.5/LKH .
RUN rm -rf workdir/LKH-3.0.5
