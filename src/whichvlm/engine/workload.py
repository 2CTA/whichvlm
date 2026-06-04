from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VisionWorkload:
    image_count: int = 1
    image_size: int = 448
    context_length: int = 4096

    def normalized(self) -> "VisionWorkload":
        return VisionWorkload(
            image_count=max(0, self.image_count),
            image_size=max(1, self.image_size),
            context_length=max(1, self.context_length),
        )
