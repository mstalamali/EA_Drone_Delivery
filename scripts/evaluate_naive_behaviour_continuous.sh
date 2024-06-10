#!/bin/bash

#Directories paths
PROJECT_HOME="${HOME}/EA_Drone_Delivery"

DATA_HOME=${PROJECT_HOME}/data

CONFIG_HOME=${PROJECT_HOME}/config

EXEC_FILE="${PROJECT_HOME}/src/ea_drone_delivery.py"

EXPERIMENT_TYPE=naive_behaviour

DATA_FOLDER="${DATA_HOME}/${EXPERIMENT_TYPE}"
mkdir -p ${DATA_FOLDER}


CONFIG_FOLDER="${CONFIG_HOME}/${EXPERIMENT_TYPE}"
mkdir -p ${CONFIG_FOLDER}

CONFIG_TEMPLATE="${CONFIG_HOME}/config_template.json"




# -------------------------------------- SET FIXED EXPERIMENT VARIABLES -------------------------------------- #
TIME_DAYS=28 # days
TIME_STEPS=$(( 24*3600*TIME_DAYS )) # seconds

SPEED=10.0 #m/s

EVAL_TYPE="continuous"

NO_EPISODES=10

MAX_BH=1.0

# LOW_THRESH=20.0 # %
# SAFETY_THRESH=5.0 # %

S_DECENTRALISED_LEARNING=0
S_CENTRALISED_LEARNING=0

ORDERS_PER_EPISODE=100

ORDER_ARRIVAL_INTERVAL=900 # s

MIN_PACKAGE_DISTANCE=2000 # m
MAX_PACKAGE_DISTANCE=8000 # m

MIN_PACKAGE_WEIGHT=1 # kg
MAX_PACKAGE_WEIGHT=5 # kg

DATA_RECORDING_INTERVAL=900 # s

INITIAL_CONFIG_TEMPLATE="${CONFIG_FOLDER}/initial_configuration.json"

sed -e "s|TIME_STEPS|${TIME_STEPS}|" \
	-e "s|EVAL_TYPE|${EVAL_TYPE}|" \
	-e "s|NO_EPISODES|${NO_EPISODES}|" \
	-e "s|SPEED|${SPEED}|" \
	-e "s|MAX_BH|${MAX_BH}|" \
	-e "s|S_DECENTRALISED_LEARNING|${S_DECENTRALISED_LEARNING}|" \
	-e "s|S_CENTRALISED_LEARNING|${S_CENTRALISED_LEARNING}|" \
	-e "s|ORDERS_PER_EPISODE|${ORDERS_PER_EPISODE}|" \
	-e "s|ORDER_ARRIVAL_INTERVAL|${ORDER_ARRIVAL_INTERVAL}|" \
	-e "s|MIN_PACKAGE_DISTANCE|${MIN_PACKAGE_DISTANCE}|" \
	-e "s|MAX_PACKAGE_DISTANCE|${MAX_PACKAGE_DISTANCE}|" \
	-e "s|MIN_PACKAGE_WEIGHT|${MIN_PACKAGE_WEIGHT}|" \
	-e "s|MAX_PACKAGE_WEIGHT|${MAX_PACKAGE_WEIGHT}|" \
	-e "s|DATA_FOLDER|${DATA_FOLDER}|" \
	-e "s|DATA_RECORDING_INTERVAL|${DATA_RECORDING_INTERVAL}|" \
	${CONFIG_TEMPLATE} > ${INITIAL_CONFIG_TEMPLATE}


# -------------------------------------- SET CHANGING EXPERIMENT VARIABLES -------------------------------------- #

NUMBER_OF_ROBOTS_LIST=(25)

MIN_BH_LIST=(0.5)

WORKING_THRESH_LIST=(50.0)

for WORKING_THRESH in ${WORKING_THRESH_LIST[*]}; do

	for S_NAIVE in ${NUMBER_OF_ROBOTS_LIST[*]}; do

		if [[ ${S_NAIVE} -eq 25 ]]; then
		    NUMBERS_OF_HOURS=02                                                                                                    
		    NUMBERS_OF_MINS=00

		elif [[ ${S_NAIVE} -eq 50 ]]; then
		    NUMBERS_OF_HOURS=04
		    NUMBERS_OF_MINS=00
		fi

		for MIN_BH in ${MIN_BH_LIST[*]}; do

			JOB_PARAM="S${S_NAIVE}_WORKING_THRESH_${WORKING_THRESH}_MINBH_${MIN_BH}_ORDER_ARRIVAL_AVERAGE${ORDER_ARRIVAL_INTERVAL}"

			CONF_FILE_PRE="${CONFIG_FOLDER}/${JOB_PARAM}"

			for SIM_NO in `seq ${1} ${2}`;
			do

				SEED=$(( 1445*SIM_NO ))

				DATA_FILE_NAME=${JOB_PARAM}_${SIM_NO}.txt

				FINAL_CONFIG_TEMPLATE="${CONF_FILE_PRE}_${SIM_NO}.json"

				sed -e "s|SEED|${SEED}|" \
					-e "s|MIN_BH|${MIN_BH}|" \
					-e "s|S_NAIVE|${S_NAIVE}|" \
					-e "s|WORKING_THRESH|${WORKING_THRESH}|" \
					-e "s|DATA_FILE_NAME|${DATA_FILE_NAME}|" \
					${INITIAL_CONFIG_TEMPLATE} > ${FINAL_CONFIG_TEMPLATE}

			done

			sed -e "s|JOBNAME|${JOB_PARAM}|" \
			-e "s|START|${1}|" \
		    -e "s|END|${2}|" \
		    -e "s|HOURS|${NUMBERS_OF_HOURS}|" \
		    -e "s|MINUTES|${NUMBERS_OF_MINS}|" \
		    run_jobarray_template.sh > run_jobarray.sh

			COMMAND="sbatch run_jobarray.sh ${EXEC_FILE} ${CONF_FILE_PRE}"
			${COMMAND}

		done
	done
done