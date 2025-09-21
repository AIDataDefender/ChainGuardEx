import logging
from logging import LoggerAdapter
import datetime
from colorama import init, Fore, Style

# Initialize colorama for cross-platform colored output
init(autoreset=True)

# Default logging configuration


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter to add colors and specific format [time][emulator][file][func][msg]
    """

    # Define colors for different log levels
    COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        """
        Format the log record with timestamp, emulator, file, function, and colored message.

        Args:
            record: The log record to format.

        Returns:
            str: Formatted log string.
        """
        # Get the color for the log level
        color = self.COLORS.get(record.levelno, Fore.WHITE)
        # Format: [YYYY-MM-DD HH:MM:SS][emulator][filename][function][message]
        log_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_format = f"[{log_time}][{record.filename}][{record.funcName}] - {color}{record.getMessage()}{Style.RESET_ALL}"
        return log_format


def setup_logger(
    log_file
) -> LoggerAdapter:
    """
    Sets up the logger with file and console handlers, using the custom colored formatter.

    Args:
        log_file (str): Path to the log file.
        log_level (int): Logging level (e.g., logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create logger
    logger = logging.getLogger(f"AppLogger_{log_file}")
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(ColoredFormatter())

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)  # Set console to WARNING level
        console_handler.setFormatter(ColoredFormatter())

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return LoggerAdapter(logger)
