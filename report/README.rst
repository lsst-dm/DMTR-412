########
DMTR-412
########

LDM-503-19a

Temporary reporting analysis while docsteady is being ported to Jira cloud.
Results used for CCR1 interim reporting. These will eventually be integrated with the VCD

Build
=====
Build the Docker image
> cd report
> docker build -t dmtr-412 .

Run coverage analysis
=====================
Run the docker image interactively and mount the local working directory.
This will open a shell in which the python coverage analysis scripts in /bin can be run

> docker run -it --rm -v  "$(pwd)":/app dmtr-412
> cd bin
> python summarizeCoverage.py

Outputs
=======
The outputs will be written to the 'output' directory




