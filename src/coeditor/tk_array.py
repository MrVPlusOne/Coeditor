from abc import ABC, abstractmethod

import numpy as np

from coeditor.common import *
from coeditor.encoding import TruncateAt, decode_tokens, truncate_section


class TkArray(ABC):
    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def tolist(self) -> TokenSeq:
        ...

    def truncate(self, dir: TruncateAt.Value, new_len: int) -> "TkArray":
        if new_len >= len(self):
            return self
        return _TruncatedTkArray(self, dir, new_len)

    @staticmethod
    def join(segs: Iterable["TkArray"], sep: int | None) -> "TkArray":
        return _JoinedTkArray(tuple(segs), sep, sum(len(seg) for seg in segs))

    @staticmethod
    def new(tks: Sequence[int]) -> "TkArray":
        return _NumpyTkArray(np.array(tks, dtype=np.int32))

    def _peek(self) -> str:
        tks = self.tolist()
        text = decode_tokens(tks)
        if len(text) > 100:
            text = text[:100] + "..."
        return text

    def __repr__(self) -> str:
        return f"TkArray(length={len(self)}, text={repr(self._peek())})"


@dataclass(frozen=True)
class _NumpyTkArray(TkArray):
    data: np.ndarray

    def __len__(self) -> int:
        return len(self.data)

    def tolist(self) -> TokenSeq:
        return self.data.tolist()


@dataclass(frozen=True)
class _JoinedTkArray(TkArray):
    "A chain-like data structure for concatenated `TkArray`s."

    segs: tuple[TkArray, ...]
    sep: int | None
    length: int

    def __len__(self) -> int:
        return self.length

    def tolist(self) -> TokenSeq:
        result = TokenSeq()
        for i, seg in enumerate(self.segs):
            if self.sep is not None and i > 0:
                result.append(self.sep)
            result.extend(seg.tolist())
        return result


@dataclass(frozen=True)
class _TruncatedTkArray(TkArray):
    "A chain-like data structure for concatenated `TkArray`s."
    original: TkArray
    direction: TruncateAt.Value
    length: int

    def __len__(self) -> int:
        return self.length

    def tolist(self) -> TokenSeq:
        return truncate_section(
            self.original.tolist(), self.direction, self.length, inplace=True
        )
