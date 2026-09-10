import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (output, dropout_pattern) as NumPy arrays matching the shape of x.
    """
    x = np.asarray(x)
    if rng is None:
        random_values = np.random.random(x.shape)
    else:
        random_values = rng.random(x.shape)

    dp_pattern = (random_values >= p).astype(float) / (1 - p)
    out = x * dp_pattern

    return (out, dp_pattern)
