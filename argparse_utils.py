import argparse

def positive_int(s: str) -> int:
    try:
        v = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f'Expected integer, got {s!r}')

    if v <= 0:
        raise argparse.ArgumentTypeError(f'Expected positive integer, got {v}')

    return v
def float_zero_to_one(s: str) -> float:
    try:
        v = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f'Expected float, got {s!r}')

    if v > 1.0 or v < 0:
        raise argparse.ArgumentTypeError(f'Expected float int the range [0, 1], got {v}')

    return v