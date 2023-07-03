import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Instantiating logging handler and record format:
handler = logging.StreamHandler()
rec_format = "%(asctime)s:%(levelname)s:%(name)s:%(funcName)s:%(message)s"
formatter = logging.Formatter(rec_format, datefmt="%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)

# Send logging handler to root logger:
logger.addHandler(handler)
