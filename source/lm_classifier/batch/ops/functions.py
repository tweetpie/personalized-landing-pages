import inspect


from typing import Callable
from typing import List


def arguments_name(f: Callable) -> List[str]:
    args = inspect.getfullargspec(f).args
    return args
