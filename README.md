# Multi-agent simulator for energy-Aware UAV-based delivery.

This repository provides a **python-based multi-agent simulator** for studying **energy-aware uav-based delivery**.

## Requirements
To be able to run the simulator, you must have:
- A recent version of Linux, MacOSX or Windows
- **Python** 3.10 or newer

## Instalation 

First clone the git repository into a given folder on your computer:
```bash
git clone https://github.com/mstalamali/EA_Drone_Delivery.git
```
Then, navigate into the newly created folder, create a new python virtual environment and install the required python packages:
```bash
cd EA_Drone_Delivery
python -m venv ea_drone_delivery
source ea_drone_delivery/bin/activate
pip install -r requirements.txt
```
## Run
First, edit the `config.json` file inside the config folder with the parameters you want to use. Then, to run the simulator, simply open a terminal, cd to the src folder and run the program.
```bash
cd path/to/src
python ea_drone_delivery.py ../config/config.json
```
## Other versions
Extended versions of the simulators are in the other branches:
- The branch **priority_queue_hybrid_bidding_with_policy_init** is the identical to the main branch but it makes it possible to initialise the robots using policies that were previously learned.
- The branch **priority_queue_hybrid_bidding_with_reservation_and_policy_init**  is the version where robots use their policies for forecasting and reserve delivery tasks for performing in the future.