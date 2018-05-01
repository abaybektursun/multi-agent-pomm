FROM python:3.6

# Set the working directory to /app
WORKDIR /pomm

# Copy the current directory contents into the container at /app
ADD . /pomm

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
RUN pip install --trusted-host pypi.python.org tensorforce
RUN pip install --trusted-host pypi.python.org tensorflow
RUN pip install --trusted-host pypi.python.org http://download.pytorch.org/whl/cu90/torch-0.4.0-cp36-cp36m-linux_x86_64.whl
RUN pip install --trusted-host pypi.python.org torchvision 

run python game.py

