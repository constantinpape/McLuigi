import logging

# TODO instantiate logger
# how to have it in all?

# FIXME this doesn't work, have a look at
#https://medium.com/google-cloud/cloud-logging-though-a-python-log-handler-a3fbeaf14704#.5ymz768mx

# could also use dict from file (YAML / JSON)
def configure_logger(logfile, loglevel = logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(logfile)
    handler.setLevel(loglevel)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
