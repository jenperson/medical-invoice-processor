import logging

def silence_noisy_loggers():
    for name in ("mistralai_workflows", "httpx", "httpcore", "temporalio"):
        logging.getLogger(name).setLevel(logging.WARNING)

