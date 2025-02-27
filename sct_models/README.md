# Supervisor Control Theory (SCT) models

The models in this project were created using [Nadzoru](https://github.com/kaszubowski/nadzoru), though they can also be imported to [Nadzoru2](https://github.com/GASR-UDESC/Nadzoru2).

## Files

- ```models/``` : Directory containing the individual capability and specification models.
- ```script.lua``` : Script that defines how to synthesize the capability and specification models together.
- ```controller.yaml``` : The UAV's control logic (i.e. supervisors).
- ```sct.py``` : Library containing functions to parse and evolve the states defined by the models in ```controller.yaml```.