##########
Quickstart
##########

In order to get started quickly two steps are necessary:
	#. Setup MySQL
    #. Install cxphasing package

1. Setup MySQL
==============
1. Edit CXParams.py - "db" section
2. Run script 'init_CXPhasing_mysql.py'. You will be prompted for super username and password to setup the database and initial tables.

2. Setup Python
===============
1. Make a copy of CXParams.py and give it a name relevant to the sample/measurement, e.g CXParams_sample1.py. 
2. Create a softlink to CXParams_sample1.py called CXParams.py ("ln -sf CXParams_sample.py CXParams.py"). The phasing scripts will always look for a file called CXParams.py so the softlink allows you to quickly change which set of parameters will be used by the algorithm.
3. In your CXParams.py edit the "io" section to have meaningul directory names for your system. Leave all the other values the same.
4. Run bin/phasing -st
5. It will complain about what python modules you need to install. I need to make a list of dependencies still.
6. It will attempt to create a log file in /var/log/CXPhasing but you will need to create that directory manually or change the log file directory in CXParams.io.log_dir.