import logging


def _configure_root_logger() -> None:
    """Configure default logging output if no handlers exist."""
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )


_configure_root_logger()

LOGGER = logging.getLogger("RTNO_v4_3")
LOGGER.setLevel(logging.INFO)
