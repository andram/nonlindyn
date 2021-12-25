from collections.abc import Iterator, Generator
import numpy as np

def local_max(generator, key = None):
    key = (lambda x:x) if key is None else key
    generator = iter(generator)
    tx_prev = next(generator)
    tx = next(generator)
    for tx_next in generator:
        if ( key(tx_prev) < key(tx)  >= key(tx_next) ): 
            yield tx
        tx_prev, tx = tx, tx_next

def local_min(generator, key = None):
    key = (lambda x:x) if key is None else key
    generator = iter(generator)
    tx_prev = next(generator)
    tx = next(generator)
    for tx_next in generator:
        if ( key(tx_prev) > key(tx)  <= key(tx_next) ): 
            yield tx
        tx_prev, tx = tx, tx_next
