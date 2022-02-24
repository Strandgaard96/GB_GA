import logging, os

# Setup logging
logger = logging
logger.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(".", "printlog.txt"), mode="w"),
        logging.StreamHandler(),
    ],
)
