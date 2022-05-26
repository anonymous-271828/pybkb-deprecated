import datetime
import logging
import time
import itertools

# Functions

def chunk_data(data, size):
    it = iter(data)
    for i in range(0, len(data), size):
        yield {k:data[k] for k in itertools.islice(it, size)}

def split_data(data, num_splits):
    out = {}
    for i, key in enumerate(data):
        if i % num_splits == 0:
            yield out
            out = {}
        out[key] = data[key]
    if out:
        yield out

# Classes

class DeltaTimeFormatter(logging.Formatter):
    def format(self, record):
        duration = datetime.datetime.utcfromtimestamp(record.relativeCreated / 1000)
        record.delta = duration.strftime("%H:%M:%S")
        return super().format(record)


