#! /bin/bash
# Interactive dialog for setting up a MILUPHCUDA run (and initial conditions for it before).
# 
# Christoph Burger 28/Dec/2017


# static stuff (which is all still asked for correctness/displayed somewhere in the interactive dialog):
SPHERESINIPATH=~/simulations/spheres_ini
SML_FACT=2.01   # sml = mean-particle-distance * sml_fact
CUDAPATH=~/simulations/cuda/miluphcuda
STATICCUDAFLAGS="-v -I rk2_adaptive -Q 1e-4 -H -a 0.5 -T 0.0 -s -g"


if [ -n "$1" ]; then    # -n tests if string has not length zero
    echo -en "\nScript to guide you through the creation of initial conditions and the subsequent start of a miluphcuda simulation run. "
    echo -e "Go to your simulation directory(!) and simply start it without any arguments.\n"
    exit
fi


echo -e "\n\nHi, I'm your personal setup routine for miluphcuda runs!"
echo "Make sure to be directly in the simulation directory ...!"
echo -en "\nWanna create initial conditions first (or do you have them already)? (y/n)  "
read CHOICE
if [ "$CHOICE" != "y" ] && [ "$CHOICE" != "n" ]; then echo -e "\nERROR. Invalid answer. Aborting ..."; exit; fi

if [ "$CHOICE" == "y" ]; then
    RELAXFLAG="TRUE"
    echo -n "Input parameter file (spheres_ini.input) is ok (y/n)? If not sure enter 'd' to display it ...  "
    read CHOICE2
    if [ "$CHOICE2" == "d" ]; then
        less spheres_ini.input
        echo -en "\nInput parameter file (spheres_ini.input) is ok (y/n)?  "
        read CHOICE2
    fi
    if [ "$CHOICE2" == "y" ]; then
        echo -n "materialconfiguration file (material.cfg) is - except for the automatically-added sml - ok (y/n)? If not sure enter 'd' to display it ...  "
        read CHOICE3
        if [ "$CHOICE3" == "d" ]; then
            less material.cfg
            echo -en "\nmaterialconfiguration file (material.cfg) is ok (y/n)?  "
            read CHOICE3
        fi
        if [ "$CHOICE3" == "y" ]; then
            echo -n "Initial conditions program binary ($SPHERESINIPATH) is ok (y/n)?  "
            read CHOICE6
            if [ "$CHOICE6" != "y" ]; then echo -e "\nAborting ..."; exit; fi
            echo -n "Do you want to start a 'solid' or a 'hydro' simulation run (s/h)?  "
            read CHOICE7
            if [ "$CHOICE7" != "s" ] && [ "$CHOICE7" != "h" ]; then echo -e "\nERROR. Invalid answer. Aborting ..."; exit; fi
            if [ "$CHOICE7" == "s" ]; then
                MATERIALMODEL="-S"
            else
                MATERIALMODEL=""
            fi
            echo -n "Do you want relaxation (via hydrostatic structures) or homogeneous (uncompressed) bodies (r/h)?  "
            read CHOICE8
            if [ "$CHOICE8" != "r" ] && [ "$CHOICE8" != "h" ]; then echo -e "\nERROR. Invalid answer. Aborting ..."; exit; fi
            if [ "$CHOICE8" == "r" ]; then
                RELAXATIONMODEL="-H"
            else
                RELAXATIONMODEL=""
            fi
            echo -n "initial distance factor (the touching-ball distance is multiplied with)?  "
            read INI_DIST_FACT
            echo -en "\nStarting initial conditions program with sml_fact = $SML_FACT ... "
            INICOMMAND="$SPHERESINIPATH $MATERIALMODEL $RELAXATIONMODEL -m material.cfg -h $SML_FACT -F $INI_DIST_FACT"
            $INICOMMAND 1>spheres_ini.log
            echo "Done!"
            echo -e "\n\n\n$INICOMMAND 1>spheres_ini.log\n" >>spheres_ini.log
        else
            echo -e "\nAborting ..."; exit
        fi
    else
        echo -e "\nAborting ..."; exit
    fi
fi




echo -e "\nLet's set up the simulation run now ..."
if [ "$RELAXFLAG" == "TRUE" ]; then
    CHOICE4="y"
else
    echo -n "Copy smoothing length from spheres_ini.log to all materials in material.cfg ... "
    SML=`grep "sml" spheres_ini.log | awk '{print $9}'`
    sed -i "/sml =/c\\\tsml = $SML;" material.cfg
    echo "Done (sml = $SML)."
    echo -n "material.cfg file is ok now (y/n)? If not sure enter 'd' to display it ...  "
    read CHOICE4
    if [ "$CHOICE4" == "d" ]; then
        less material.cfg
        echo -en "\nmaterial.cfg is ok (y/n)?  "
        read CHOICE4
    fi
fi

if [ "$CHOICE4" == "y" ]; then
    echo -e "\n`grep "courant-like" spheres_ini.log`"
    echo -n "Overall number of timesteps to compute?  "
    read NSTEPS
    echo -n "Length of output timestep?  "
    read DT
    echo -n "Maximum length of the (Runge Kutta) timestep?  "
    read MAXDT
    N=`cat impact.0000 | wc -l`
    
    CUDACOMMAND="$CUDAPATH $STATICCUDAFLAGS -N $N -n $NSTEPS -t $DT -M $MAXDT -f impact.0000 -m material.cfg"
    
    echo -en "\nIs the miluphcuda commandline\n\t$CUDACOMMAND\nok (y/n)?  "
    read CHOICE5
    if [ "$CHOICE5" == "y" ]; then
        echo -e "Starting simulation run ...\n"
        echo -e "time( $CUDACOMMAND 1> miluphcuda.output 2>miluphcuda.error ) >>spheres_ini.log & disown -h\n\n" >>spheres_ini.log
        echo "Simulation started at: `date`" >>spheres_ini.log
        echo "On host: `hostname`" >>spheres_ini.log
        time( $CUDACOMMAND 1> miluphcuda.output 2>miluphcuda.error ) >>spheres_ini.log & disown -h
    else
        echo -e "\nAborting ..."; exit
    fi
else
    echo -e "\nAborting ..."; exit
fi

