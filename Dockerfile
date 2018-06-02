FROM python:3.6

# Set the working directory to /app
WORKDIR /pomm

# Copy the current directory contents into the container at /app
ADD . /pomm

EXPOSE 10080

# Update aptitude with new repo
RUN apt-get update

# Install software 
RUN apt-get install -y git 

# Clone the conf files into the docker container
RUN git clone https://github.com/MultiAgentLearning/playground

# Install Pommerman
#RUN cd /playground && pip install .
RUN pip install ./playground/.

RUN pip install --trusted-host pypi.python.org -r ./playground/requirements.txt
RUN pip install --trusted-host pypi.python.org -r ./playground/requirements_extra.txt
RUN pip install --trusted-host pypi.python.org tensorflow
RUN pip install --trusted-host pypi.python.org matplotlib
#RUN apt-get install -y python3-matplotlib

COPY . /pomm
# Run app.py when the container launches
WORKDIR /pomm/RNN_agent

ENTRYPOINT ["python"]
CMD ["run.py"]


