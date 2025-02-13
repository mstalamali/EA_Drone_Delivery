#!/bin/bash

#Directories paths
PROJECT_HOME="${HOME}/EA_Drone_Delivery"

DATA_HOME=${PROJECT_HOME}/data

CONFIG_HOME=${PROJECT_HOME}/config

EXEC_FILE="${PROJECT_HOME}/src/ea_drone_delivery.py"

EXPERIMENT_TYPE=learning_based_distance_modified_huber_behaviour

CONFIG_TEMPLATE="${CONFIG_HOME}/config_template.json"

# -------------------------------------- SET FIXED EXPERIMENT VARIABLES -------------------------------------- #
TIME_DAYS=56 # days
TIME_STEPS=$(( 24*3600*TIME_DAYS )) # seconds

SPEED=10.0 #m/s

EVAL_TYPE="continuous"

NO_EPISODES=10

MAX_BH=1.0

# LOW_THRESH=20.0 # %
# SAFETY_THRESH=5.0 # %

S_NAIVE=0
S_DECENTRALISED_LEARNING_HEURISTIC=0
S_DECENTRALISED_LEARNING_PROBABILITY=0

ORDERS_PER_EPISODE=100

# ORDER_ARRIVAL_INTERVAL=900 # s

MIN_PACKAGE_DISTANCE=1000 # m
MAX_PACKAGE_DISTANCE=8000 # m

MIN_PACKAGE_WEIGHT=0.5 # kg
MAX_PACKAGE_WEIGHT=5.0 # kg

DATA_RECORDING_INTERVAL=900 # s

WORKING_THRESH=50.0

MIN_BH=0.5

LOOK_AHEAD_SIZE=10

TIMEOUT=3600

LOSS_FUNCTION=modified_huber

BIDDING_STRATEGY=WeakPrioritisation

MODEL_INIT_METHOD=AssumptionFitting # AssumptionFitting, CanDoEverything, DataGathering

SCALER_INIT_METHOD=AssumptionMeanVariance # KnownMeanVariance, AssumptionMeanVariance, DataMeanVariance

SCALER_TYPE="Standard" # Standard , MinMax

AGENTS_DATA_LOGGING=false

DATA_AUGMENTATION_PTS=0

DATA_FOLDER="${DATA_HOME}/${EXPERIMENT_TYPE}/${SCALER_TYPE}_${MODEL_INIT_METHOD}_${SCALER_INIT_METHOD}_${BIDDING_STRATEGY}"
mkdir -p ${DATA_FOLDER}


CONFIG_FOLDER="${CONFIG_HOME}/${EXPERIMENT_TYPE}/${SCALER_TYPE}_${MODEL_INIT_METHOD}_${SCALER_INIT_METHOD}_${BIDDING_STRATEGY}"
mkdir -p ${CONFIG_FOLDER}

INITIAL_CONFIG_TEMPLATE="${CONFIG_FOLDER}/initial_configuration.json"

sed -e "s|TIME_STEPS|${TIME_STEPS}|" \
	-e "s|EVAL_TYPE|${EVAL_TYPE}|" \
	-e "s|NO_EPISODES|${NO_EPISODES}|" \
	-e "s|SPEED|${SPEED}|" \
	-e "s|MAX_BH|${MAX_BH}|" \
	-e "s|S_NAIVE|${S_NAIVE}|" \
	-e "s|S_DECENTRALISED_LEARNING_HEURISTIC|${S_DECENTRALISED_LEARNING_HEURISTIC}|" \
	-e "s|S_DECENTRALISED_LEARNING_PROBABILITY|${S_DECENTRALISED_LEARNING_PROBABILITY}|" \
	-e "s|ORDERS_PER_EPISODE|${ORDERS_PER_EPISODE}|" \
	-e "s|MIN_PACKAGE_DISTANCE|${MIN_PACKAGE_DISTANCE}|" \
	-e "s|MAX_PACKAGE_DISTANCE|${MAX_PACKAGE_DISTANCE}|" \
	-e "s|MIN_PACKAGE_WEIGHT|${MIN_PACKAGE_WEIGHT}|" \
	-e "s|MAX_PACKAGE_WEIGHT|${MAX_PACKAGE_WEIGHT}|" \
	-e "s|DATA_FOLDER|${DATA_FOLDER}|" \
	-e "s|MIN_BH|${MIN_BH}|" \
	-e "s|WORKING_THRESH|${WORKING_THRESH}|" \
	-e "s|LOOK_AHEAD_SIZE|${LOOK_AHEAD_SIZE}|" \
	-e "s|TIMEOUT|${TIMEOUT}|" \
	-e "s|DATA_RECORDING_INTERVAL|${DATA_RECORDING_INTERVAL}|" \
	-e "s|LOSS_FUNCTION|${LOSS_FUNCTION}|" \
	-e "s|SCALER_TYPE|${SCALER_TYPE}|" \
	-e "s|BIDDING_STRATEGY|${BIDDING_STRATEGY}|" \
	-e "s|MODEL_INIT_METHOD|${MODEL_INIT_METHOD}|" \
	-e "s|SCALER_INIT_METHOD|${SCALER_INIT_METHOD}|" \
	-e "s|AGENTS_DATA_LOGGING|${AGENTS_DATA_LOGGING}|" \
	-e "s|DATA_AUGMENTATION_PTS|${DATA_AUGMENTATION_PTS}|" \
	${CONFIG_TEMPLATE} > ${INITIAL_CONFIG_TEMPLATE}

# -------------------------------------- SET CHANGING EXPERIMENT VARIABLES -------------------------------------- #
INIT_PTS_LIST=(0)

# BIDDING_STRATEGY_LIST=(\"strong_prioritisation\" \"random\")

NUMBER_OF_ROBOTS_LIST=(25)

ORDER_ARRIVAL_INTERVAL_LIST=(600 900)

EXPLORATION_PROBABILITY_LIST=(0.0)

for INIT_PTS in ${INIT_PTS_LIST[*]}; do

	for ORDER_ARRIVAL_INTERVAL in ${ORDER_ARRIVAL_INTERVAL_LIST[*]}; do

		# echo ${ORDER_ARRIVAL_INTERVAL}

		for S_DECENTRALISED_LEARNING_DISTANCE in ${NUMBER_OF_ROBOTS_LIST[*]}; do

			# echo ${S_DECENTRALISED_LEARNING_DISTANCE}

			if [[ ${S_DECENTRALISED_LEARNING_DISTANCE} -eq 25 ]]; then

				if [[ ${TIME_DAYS} -eq 56 ]]; then
			    	NUMBERS_OF_HOURS=12
			    elif [[ ${TIME_DAYS} -eq 28 ]]; then
			    	NUMBERS_OF_HOURS=06
			    fi

			    NUMBERS_OF_MINS=00

			elif [[ ${S_DECENTRALISED_LEARNING_DISTANCE} -eq 50 ]]; then

				if [[ ${TIME_DAYS} -eq 56 ]]; then
			    	NUMBERS_OF_HOURS=24
			    elif [[ ${TIME_DAYS} -eq 28 ]]; then
			    	NUMBERS_OF_HOURS=12
			    fi

			    NUMBERS_OF_MINS=00
			fi

			for EXPLORATION_PROBABILITY in ${EXPLORATION_PROBABILITY_LIST[*]}; do

				# echo ${EXPLORATION_PROBABILITY}

				JOB_PARAM="S${S_DECENTRALISED_LEARNING_DISTANCE}_WORKING_THRESH_${WORKING_THRESH}_MINBH_${MIN_BH}_ORDER_ARRIVAL_AVERAGE${ORDER_ARRIVAL_INTERVAL}_EXPLORATION_PROBABILITY_${EXPLORATION_PROBABILITY}_INIT_PTS_${INIT_PTS}_DATA_AUGMENTATION_PTS_${DATA_AUGMENTATION_PTS}"

				# echo ${JOB_PARAM}

				CONF_FILE_PRE="${CONFIG_FOLDER}/${JOB_PARAM}"

				for SIM_NO in `seq ${1} ${2}`;
				do

					SEED=$(( 1445*SIM_NO ))

					DATA_FILE_NAME=${JOB_PARAM}_${SIM_NO}.txt

					FINAL_CONFIG_TEMPLATE="${CONF_FILE_PRE}_${SIM_NO}.json"

					sed -e "s|SEED|${SEED}|" \
						-e "s|EXPLORATION_PROBABILITY|${EXPLORATION_PROBABILITY}|" \
						-e "s|S_DECENTRALISED_LEARNING_DISTANCE|${S_DECENTRALISED_LEARNING_DISTANCE}|" \
						-e "s|INIT_PTS|${INIT_PTS}|" \
						-e "s|DATA_FILE_NAME|${DATA_FILE_NAME}|" \
						-e "s|ORDER_ARRIVAL_INTERVAL|${ORDER_ARRIVAL_INTERVAL}|" \
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
done