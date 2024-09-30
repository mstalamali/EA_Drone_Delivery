import time
from controllers.main_controller import MainController, Configuration
from controllers.view_controller import ViewController
from multiprocessing import Pool
from pathlib import Path
from os.path import join
from sys import argv

import time


def main():
    
    # open the config file where the simulation params are specified
    config = Configuration(config_file=argv[1])

    # create controller object
    main_controller = MainController(config)

    # run simulation with visualisation if this one is activated in config file
    if config.value_of("visualization")['activate']:    
        view_controller = ViewController(main_controller,
                                         config.value_of("width"),
                                         config.value_of("height"),
                                         config.value_of("visualization")['fps'],
                                         config.value_of("visualization")['graphics'])

    # run simulation without visuation if this one is not activated in config file
    else:
        main_controller.start_simulation()

if __name__ == '__main__':
    main()
