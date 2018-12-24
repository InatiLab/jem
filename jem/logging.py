import logging


logger = logging.getLogger(__name__)

# To show debug messages, uncomment this
# from outside the module, import the logger object and use something like this
# logger.setLevel(logging.DEBUG)

# add a console handler
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
