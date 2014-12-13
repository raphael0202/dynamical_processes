# -*- coding: utf8 -*-

import logging
from logging.handlers import RotatingFileHandler
import json
import os


def start_logging(logger_level="DEBUG", file_handler_level="INFO", steam_handler_level="DEBUG", log_file=False):
    # création de l'objet logger qui va nous servir à écrire dans les logs
    logger = logging.getLogger()
    # on met le niveau du logger à DEBUG, comme ça il écrit tout
    
    if logger_level == "DEBUG":
        logger.setLevel(logging.DEBUG)
    elif logger_level == "INFO":
        logger.setLevel(logging.INFO)
    elif logger_level == "WARNING":
        logger.setLevel(logging.WARNING)
    else:
        raise ValueError("Unknown level argument: {}".format(logger_level))

    # création d'un formateur qui va ajouter le temps, le niveau
    # de chaque message quand on écrira un message dans le log
    formatter_file = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
    formatter_steam = logging.Formatter('%(levelname)s :: %(message)s')

    if log_file:
        # création d'un handler qui va rediriger une écriture du log vers
        # un fichier en mode 'append', avec 1 backup et une taille max de 1Mo
        file_handler = RotatingFileHandler('activity.log', 'a', 1000000, 1)
        # on lui met le niveau sur DEBUG, on lui dit qu'il doit utiliser le formateur
        # créé précédement et on ajoute ce handler au logger

        if file_handler_level == "DEBUG":
            file_handler.setLevel(logging.DEBUG)
        elif file_handler_level == "INFO":
            file_handler.setLevel(logging.INFO)
        elif file_handler_level == "WARNING":
            file_handler.setLevel(logging.WARNING)
        else:
            raise ValueError("Unknown level argument: {}".format(file_handler_level))

        file_handler.setFormatter(formatter_file)
        logger.addHandler(file_handler)
        
    # création d'un second handler qui va rediriger chaque écriture de log
    # sur la console
    steam_handler = logging.StreamHandler()
    
    if steam_handler_level == "DEBUG":
        steam_handler.setLevel(logging.DEBUG)
    elif steam_handler_level == "INFO":
        steam_handler.setLevel(logging.INFO)
    elif steam_handler_level == "WARNING":
        steam_handler.setLevel(logging.WARNING)
    else:
        raise ValueError("Unknown level argument: {}".format(steam_handler_level))
    
    steam_handler.setFormatter(formatter_steam)
    logger.addHandler(steam_handler)

    return logger


def save_json(data, filename):
    if not os.path.isdir("data"):
        os.mkdir("data")

    with open("data/{}.json".format(filename), 'w') as fp:
        logging.info("Saving JSON file: {}".format(filename))
        json.dump(data, fp)
