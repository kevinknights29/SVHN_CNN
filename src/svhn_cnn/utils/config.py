import os
import configparser
from pathlib import Path


UTILS_DIR = os.path.dirname(__file__)
MODULE_DIR = os.path.dirname(UTILS_DIR)
SOURCE_DIR = os.path.dirname(MODULE_DIR)
PROJECT_DIR = os.path.dirname(SOURCE_DIR)
CONFIG_FILEPATH = Path(PROJECT_DIR) / "config.ini"

# TODO: Convert into a singleton
config = configparser.ConfigParser()
config.read(CONFIG_FILEPATH)
