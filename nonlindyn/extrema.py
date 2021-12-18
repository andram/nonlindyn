from collections.abc import Iterator, Generator
import numpy as np

def filter_max(
        generator: Iterator[tuple[float,np.ndarray]],
        index: int = 0
) -> Generator[tuple[float,np.ndarray], None, None]:
    tx_old = next(generator)
    tx = next(generator)
    for tx_next in generator:
        if (
                tx_old[1][index] < tx[1][index]
                and tx[1][index] >= tx_next[1][index]
        ): # local maximum
            # TODO: proper interpolation
            yield tx
        tx_old = tx
        tx = tx_next
