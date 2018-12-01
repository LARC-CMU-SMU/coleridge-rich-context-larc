from ubuntu:18.04

# Run apt to install OS packages
RUN apt update
RUN apt install -y tree vim curl python3 python3-pip git

# Requirements for Science Parse

RUN apt-get -y install python3-software-properties software-properties-common debconf-utils

RUN add-apt-repository -y ppa:webupd8team/java && \
    apt-get update && \
    echo "oracle-java8-installer shared/accepted-oracle-license-v1-1 select true" | debconf-set-selections && \
    apt-get -y install oracle-java8-installer

# Python 3 package install example
RUN pip3 install ipython matplotlib numpy pandas scikit-learn scipy six nltk tqdm
RUN python3 -c "import nltk; nltk.download(['punkt'])"

# Use utf-8 encoding
ENV PYTHONIOENCODING=utf-8

# create directory for "work".
RUN mkdir /work

# clone the rich context repo into /rich-context-competition
RUN git clone https://github.com/Coleridge-Initiative/rich-context-competition.git /rich-context-competition

LABEL maintainer="jonathan.morgan@nyu.edu"
