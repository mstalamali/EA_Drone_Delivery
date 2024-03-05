import time

# import pandas as pd

from controllers.main_controller import MainController, Configuration
from controllers.view_controller import ViewController
from multiprocessing import Pool
from pathlib import Path
from os.path import join
from sys import argv

import time


def main():
    config = Configuration(config_file=argv[1])

    main_controller = MainController(config)

    if config.value_of("visualization")['activate']:    
        view_controller = ViewController(main_controller,
                                         config.value_of("width"),
                                         config.value_of("height"),
                                         config.value_of("visualization")['fps'],
                                         config.value_of("visualization")['graphics'])

    else:
        main_controller.start_simulation()

if __name__ == '__main__':
    # start_time = time.time()
    main()
    # print("--- %s seconds ---" % (time.time() - start_time))
