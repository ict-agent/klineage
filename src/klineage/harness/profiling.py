"""NCU capture options."""

import re
from dataclasses import dataclass

_SETS = frozenset({"basic", "detailed", "full"})
_SECTION = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


@dataclass(frozen=True, slots=True)
class ProfileOptions:
    set: str = "detailed"
    sections: tuple[str, ...] = ()
    kernel_filter: str | None = None
    timeout_seconds: int = 180

    def __post_init__(self):
        if self.set not in _SETS:
            raise ValueError("unknown NCU set")
        sections = tuple(self.sections)
        if any(not isinstance(s, str) or not _SECTION.fullmatch(s) for s in sections):
            raise ValueError("invalid NCU section identifier")
        object.__setattr__(self, "sections", sections)
        if self.kernel_filter is not None:
            if (
                not isinstance(self.kernel_filter, str)
                or not self.kernel_filter.strip()
            ):
                raise ValueError("kernel_filter must be a nonempty regex")
            re.compile(self.kernel_filter)
        if type(self.timeout_seconds) is not int or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive integer")


__all__ = ["ProfileOptions"]
