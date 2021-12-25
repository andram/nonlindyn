from collections.abc import Iterator, Generator
import itertools as it

def local_max(generator, key = None):
    """
    Calculate the local maximum
    """
    key = (lambda x:x) if key is None else key
    generator = iter(generator)
    tx_prev, tx = next(generator), next(generator) 
    for tx_next in generator:
        if ( key(tx_prev) < key(tx)  >= key(tx_next) ): 
            yield tx
        tx_prev, tx = tx, tx_next
        
def local_min(generator, key = None):
    """
    Calculate the local minimum
    """
    # Let's just re-use the local maximum function with inverted logic. 
    key_inv = (lambda x:-x) if key is None else (lambda x: -key(x))
    return local_max(generator, key=key_inv)
