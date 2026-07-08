from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


WorkloadTask = Literal[
    "image_qa",
    "ocr",
    "document",
    "chart",
    "video",
    "audio",
    "general_multimodal",
]


@dataclass(frozen=True)
class Workload:
    task: WorkloadTask = "image_qa"
    context_length: int = 4096
    image_count: int = 0
    image_size: int = 448
    video_frames: int = 0
    audio_seconds: float = 0.0
    batch_size: int = 1

    def normalized(self) -> "Workload":
        return Workload(
            task=self.task,
            context_length=max(1, self.context_length),
            image_count=max(0, self.image_count),
            image_size=max(1, self.image_size),
            video_frames=max(0, self.video_frames),
            audio_seconds=max(0.0, self.audio_seconds),
            batch_size=max(1, self.batch_size),
        )


@dataclass(frozen=True)
class VisionWorkload(Workload):
    image_count: int = 1
