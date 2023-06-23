import os
import psutil

"""
    CATS memory management
"""

_MAX_MEMORY_FOR_CATS_NAME = "CATS_MAX_MEMORY_USAGE"


def set_max_memory_for_cats(memory_in_bytes: int):
    """
        Environment variable.
        Specifies max memory in bytes that CATS can use.
    """
    os.environ[_MAX_MEMORY_FOR_CATS_NAME] = str(memory_in_bytes)


def get_max_memory_for_cats():
    """
        Environment variable.
        Returns max memory in bytes that CATS can use.
    """
    return float(os.environ[_MAX_MEMORY_FOR_CATS_NAME])


set_max_memory_for_cats(-1)  # `-1` means that maximum available for cats


_MIN_MEMORY_NOT_USED_BY_CATS_NAME = "MIN_MEMORY_NOT_USED_BY_CATS"


def set_min_memory_not_used_by_cats(memory_in_bytes: int):
    """
        Environment variable.
        Specifies min memory in bytes that CATS cannot use.
    """
    os.environ[_MIN_MEMORY_NOT_USED_BY_CATS_NAME] = str(memory_in_bytes)


def get_min_memory_not_used_by_cats():
    return float(os.environ[_MIN_MEMORY_NOT_USED_BY_CATS_NAME])


set_min_memory_not_used_by_cats(1024**3 / 4)  # min 0.25 GB will not be used by CATS


def get_max_memory_available_for_cats():
    """
        Returns memory in bytes that CATS can use taking into account max allowed, max available,
        and memory that cannot be used.
    """
    min_reserved = get_min_memory_not_used_by_cats()
    max_allowed = get_max_memory_for_cats()
    max_available = psutil.virtual_memory().available - min_reserved
    available_for_cats = max_available if (max_allowed == -1) else min(max_allowed, max_available)
    if available_for_cats < 0:
        raise MemoryError("Not enough available memory (RAM) for CATS. Try setting higher allowed memory "
                          "`cats.env_variables.set_cats_max_memory_usage(memory_in_bytes)` "
                          "or consider reducing memory not used by CATS "
                          "`cats.env_variables.set_min_memory_not_used_by_cats(memory_in_bytes)`, "
                          "otherwise, close some apps to free up RAM.")
    return available_for_cats


"""
    B-E-DATE block sizing
"""

_MIN_BEDATE_BLOCK_SIZE_NAME = "MIN_BEDATE_BLOCK_SIZE"


def set_min_bedate_block_size(size: int):
    """
        Environment variable.
        Specifies min time frame size for B-E-DATE.
    """
    os.environ[_MIN_BEDATE_BLOCK_SIZE_NAME] = str(size)


def get_min_bedate_block_size():
    """
        Environment variable.
        Returns min time frame size for B-E-DATE.
    """
    return int(os.environ[_MIN_BEDATE_BLOCK_SIZE_NAME])


set_min_bedate_block_size(250)  # at least 250 points
