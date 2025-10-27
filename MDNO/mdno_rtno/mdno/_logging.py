import logging


def _configure_root_logger() -> None:
    """Configure default logging if no handlers are present."""
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )


_configure_root_logger()

LOGGER = logging.getLogger("MDNO_v5_3")
LOGGER.setLevel(logging.INFO)
