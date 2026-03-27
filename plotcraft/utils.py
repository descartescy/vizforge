import math

def floor_significant_digits(x:int | float, digits:int) -> int | float:
    """Round a number DOWN to the specified number of significant digits.

    This function always rounds toward negative infinity
    to retain a fixed number of significant digits,
    without rounding up. This is especially useful for
    truncating values strictly downward for numerical precision.

    Parameters
    ----------
    x : int or float
        Input number to be rounded down to significant digits.

    digits : int
        Number of significant digits to retain.
        Must be a positive integer.

    Returns
    -------
    int or float
        The input value rounded down to the specified
        number of significant digits.

    Raises
    ------
    ValueError
        If ``digits`` is not a positive integer.

    Examples
    --------
    >>> floor_significant_digits(123456, 2)
    120000
    >>> floor_significant_digits(-123456, 2)
    -120000
    >>> floor_significant_digits(1.23456, 2)
    1.2
    >>> floor_significant_digits(-1.23456, 2)
    -1.2
    """
    if digits <= 0 or type(digits) != int:
        raise ValueError("floor significant digits should be positive int")
    if x == 0:
        return 0
    elif x > 0:
        exp = math.floor(math.log10(x))
        decimals = exp - digits + 1
        if decimals < 0:
            decimals = -decimals
            scale = 10 ** decimals
            return math.floor(x * scale) / scale
        else:
            scale = 10 ** decimals
            return math.floor(x / scale) * scale
    else:
        x = abs(x)
        return -floor_significant_digits(x,digits)