import logging.config


def configure_logging(filename=None):
    time_format = '[%(asctime)s] :: %(name)s - %(levelname)s - %(message)s'
    simple_format = '%(name)s - %(message)s'

    logging_config = {
        'version': 1,
        'formatters': {
            'time': {'format': time_format},
            'simple': {'format': simple_format},
            },
        'handlers': {
            # This is what gets printed out to the console
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'simple',
                'level': 'DEBUG',
                },
            },
        'loggers': {
            '__main__': {'level': 'DEBUG'},
            'cochlear': {'level': 'DEBUG'},
            'cochlear.nidaqmx': {'level': 'ERROR'},
            },
        'root': {
            'handlers': ['console'],
            },
        }
    if filename is not None:
        logging_config['handlers']['file'] = {
            'class': 'logging.FileHandler',
            'formatter': 'time',
            'filename': filename,
            'level': 'DEBUG',
        }
        logging_config['root']['handlers'].append('file')
    logging.config.dictConfig(logging_config)
