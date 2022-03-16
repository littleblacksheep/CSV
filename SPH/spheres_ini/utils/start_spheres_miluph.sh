#! /bin/bash

# This script creates initial conditions (spheres) for use with miluph ($INICOMMAND)
# and starts the simulation afterwards ($MILUPHCOMMAND).
# Christoph Burger 28/Dec/2017

# Before starting this script a folder taking the simulation result files should be prepared.
# This folder has to contain the input parameter file (usually simulation.input).
# This folder is passed as the first command line argument.
# The script can be started from any directory.


# set paths and filenames for the program for generating initial conditions and for miluph:
INIPATH=~/simulations
MILUPHPATH=~/simulations/miluph150915_solid
ININAME=spheres_ini
MILUPHNAME=miluph


if [ "$1" == "?" ] || [ "$1" == "-?" ]; then
	echo "This script creates initial conditions (spheres) for use with miluph and starts the simulation run afterwards."
	echo "  Usage: $0 \$1 \$2 \$3 \$4 \$5"
	echo "    All arguments mandatory, except for \$5:"
	echo "        \$1    simulation directory (where input parameter file (usually simulation.input) must be located)"
	echo "        \$2    sml_factor (sml = mean-particle-distance * sml_factor)"
	echo "        \$3    number of miluph timesteps"
	echo "        \$4    length of single timestep"
	echo "        \$5    initial distance factor (the \"touching ball\" distance is multiplied with)"
	echo "Assumed location of the initial conditions program is: $INIPATH/$ININAME"
	echo "And of miluph: $MILUPHPATH/$MILUPHNAME"
	exit
fi


# convert simulation directory to absolute path:
SIMPATH=`cd $1; pwd`

# set files (INFILE refers to the program for initial conditions):
INFILE=$SIMPATH/simulation.input
INIFILE=$SIMPATH/impact.0000
MATFILE="$MILUPHPATH/materialconstants.data"
SCENFILE=$SIMPATH/materialscenario.data

SIMLOGFILE=$SIMPATH/simulation.logfile	#logfile for the initial configuration program and for this script


# build commandlines for initial configuration program and miluph:
INICOMMAND="$INIPATH/$ININAME -S -H -f $INFILE -o $INIFILE -m $MATFILE -s $SCENFILE -h $2"
if [ -n "$5" ];then	#-n means string has not zero length
	INICOMMAND="$INICOMMAND -F $5"
fi
MILUPHCOMMAND="$MILUPHPATH/$MILUPHNAME -m $MATFILE -s $SCENFILE -f $INIFILE -v -a 0.5 -g -n $3 -t $4 -Q 1e-5"


# building initial particle configuration:
echo "========  Building initial particle configuration ...  ========" > $SIMLOGFILE
$INICOMMAND >> $SIMLOGFILE 2>> $SIMLOGFILE
if [ "$?" != "0" ]; then
	echo "ERROR during generation of initial particle distribution! Check $SIMLOGFILE!"
	exit
fi
echo "========  Building initial particle configuration completed.  ========" >> $SIMLOGFILE


# starting simulation run with miluph:
echo "========  Starting simulation run with miluph ...  ========" >> $SIMLOGFILE
echo "Command line for the start script was:  "$0" "$* >> $SIMLOGFILE
echo "Command line for initial configuration program was:  "$INICOMMAND >> $SIMLOGFILE
echo "Command line for miluph will be:  "$MILUPHCOMMAND >> $SIMLOGFILE
echo -e "\nTiming start: "`date` >> $SIMLOGFILE
time ( $MILUPHCOMMAND > $SIMPATH/miluph.output 2> $SIMPATH/miluph.error ) &>> $SIMLOGFILE
echo -e "\nTiming end:   "`date` >> $SIMLOGFILE
echo "========  miluph simulation run completed. BUT anyway take a look at $SIMPATH/miluph.error - should be an empty file!  ========" >> $SIMLOGFILE



