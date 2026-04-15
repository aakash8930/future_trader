import asyncio
import json
import os
import time
from pathlib import Path
from typing import AsyncGenerator

from sse_starlette import EventSourceResponse

EVENTS_PATH = Path("logs/events.jsonl")


async def event_generator() -> AsyncGenerator:
    """Yield new lines from events.jsonl as SSE events using seek(0,2) tail approach."""
    while True:
        try:
            if not EVENTS_PATH.exists():
                await asyncio.sleep(1)
                continue

            with open(EVENTS_PATH, "r") as f:
                f.seek(0, 2)  # seek to end
                while True:
                    line = f.readline()
                    if line:
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                yield {"event": "message", "data": json.dumps(data)}
                            except json.JSONDecodeError:
                                yield {"event": "message", "data": json.dumps({"message": line})}
                    else:
                        await asyncio.sleep(1)
        except Exception:
            await asyncio.sleep(1)


async def stream_events(request):
    """SSE endpoint that streams events.jsonl to the client."""
    async def _inner():
        async for event in event_generator():
            yield event

    return EventSourceResponse(
        _inner(),
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
