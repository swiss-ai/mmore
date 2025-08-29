import logging

# Create or get the logger

WEBSRCH_EMOJI = "🌐"
logger = logging.getLogger("WEBSEARCHRAG")
logging.basicConfig(
    format=f"[WebSearch {WEBSRCH_EMOJI} -- %(asctime)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Prevent multiple handlers if the logger is configured multiple times
if not logger.handlers:
    # Create a file handler to log to a file
    file_handler = logging.FileHandler("shared_log_file.log")
    file_handler.setLevel(logging.DEBUG)

    # Define log format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # Add file handler to logger
    logger.addHandler(file_handler)
