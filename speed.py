from typing import Any, Collection, Sequence, TypeVar, Union

print(isinstance(tuple(), Collection))
print(isinstance(tuple(), Sequence))
print(isinstance(iter(tuple()), Collection))
print(isinstance(iter(tuple()), Sequence))
