"""Websocket bridge: a recorded trial -> the browser, at 20 Hz.

The simulation is **precomputed, then streamed**, and that is a deliberate choice rather
than a shortcut. A trial is 8,000 timesteps over 165k neurons and takes several seconds to
compute, so it cannot be produced at wall-clock speed; and a giant fiber escape lasts a few
milliseconds, which at real time would fall inside a single 50 ms frame and be invisible
anyway. Both facts point the same way: compute once, then play back under an adjustable
time dilation. That also gives seeking and instant replay for free.

Colour updates go out at 20 Hz. Nothing is ever sent per simulation timestep.

The payload carries smoothed activity, not spikes - see :mod:`sim.recorder` for why, and for
what "1.0" means.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

RENDER_HZ = 20.0

#: Simulation milliseconds per wall-clock millisecond. The escape itself lasts a few ms, so
#: the useful range is heavily slowed; 1.0 would be real time and show nothing.
DEFAULT_SPEED = 0.02


@dataclass
class Playback:
    """Mutable playback state, shared by every connected client."""

    step: int = 0
    playing: bool = True
    speed: float = DEFAULT_SPEED
    loop: bool = True


@dataclass
class DemoState:
    """What the server is currently able to stream."""

    recording: Any
    aggregates: dict[str, np.ndarray]
    palette: dict[str, str]
    display_scale: float = 1.0
    #: Physics for the same trial: fly and predator paths on the neural clock, plus the
    #: outcome. Sent once with hello rather than per frame - the client already knows the
    #: current step, so it can index the paths itself.
    world: dict[str, Any] | None = None
    playback: Playback = field(default_factory=Playback)
    busy: str | None = None
    rerun: Any = None  # callable(ratio_ms, gain_scale, seed) -> (Recording, event)


def display_scale_for(activity: np.ndarray) -> float:
    """Auto-range the stream, the way a microscope display LUT does.

    Necessary because the dynamic range across the pathway is enormous: at a plausible
    encoder gain the LC populations settle around 0.002 while the giant fiber reaches 0.45,
    a ratio of over 200. A fixed full-scale would render the LC bundles black.

    A high percentile rather than the maximum, so one briefly-saturating cell does not
    compress everything else. The chosen scale is reported to the client so the display can
    say what full brightness corresponds to - the transform is made visible, not hidden.
    """
    if activity.size == 0:
        return 1.0
    return float(max(np.percentile(activity, 99.9), 1e-6))


def _quantise(values: np.ndarray, scale: float) -> list[int]:
    """Activity to 0-255 against the auto-ranged scale."""
    return np.clip(values / scale * 255.0, 0.0, 255.0).astype(np.uint8).tolist()


def build_app(state: DemoState) -> FastAPI:
    app = FastAPI(title="brain-view stream")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    def history(stride: int = 8) -> dict:
        """The whole trial's traces, decimated, sent once.

        The trace panel used to accumulate samples as frames arrived, which meant it was
        blank for the first several seconds and could never show anything behind the point
        where the client happened to connect. The server already holds the entire recording,
        so it just sends it: the panel then draws a window ending at the playhead, populated
        from the first frame.
        """
        recording = state.recording
        scene = recording.scene
        picks = np.arange(0, recording.n_steps, stride)
        payload: dict[str, Any] = {
            "stride": stride,
            "aggregate": {
                name: np.round(values[picks], 5).tolist()
                for name, values in state.aggregates.items()
            },
        }
        if scene is not None:
            payload["theta_deg"] = np.round(scene.theta_deg[picks], 3).tolist()
            payload["theta_dot"] = np.round(scene.theta_dot_deg_per_ms[picks], 5).tolist()
        return payload

    def world_payload() -> dict | None:
        """Decimate the physics paths onto the same stride the traces use."""
        if state.world is None:
            return None
        stride = 8
        payload = dict(state.world)
        for key in ("fly", "predator"):
            path = np.asarray(payload[key])
            payload[key] = np.round(path[::stride], 4).tolist()
        payload["stride"] = stride
        return payload

    def hello() -> dict:
        recording = state.recording
        scene = recording.scene
        return {
            "history": history(),
            "world": world_payload(),
            "type": "hello",
            "body_ids": [str(b) for b in recording.body_ids],
            "cell_types": recording.cell_types,
            "aggregate_types": sorted(state.aggregates),
            "palette": state.palette,
            "dt_ms": recording.dt_ms,
            "duration_ms": recording.duration_ms,
            "n_steps": recording.n_steps,
            "render_hz": RENDER_HZ,
            "speed": state.playback.speed,
            "collision_ms": None if scene is None else scene.collision_ms,
            "ratio_ms": None if scene is None else scene.ratio_ms,
            "events": recording.events,
            "meta": {
                "dataset": recording.meta.get("dataset"),
                "gain_scale": recording.meta.get("trial", {}).get("gain_scale"),
                "activity_tau_ms": recording.meta.get("activity_tau_ms"),
                "reference_rate_hz": recording.meta.get("reference_rate_hz"),
                "display_scale": state.display_scale,
                "display_full_scale_hz": state.display_scale
                * float(recording.meta.get("reference_rate_hz", 150.0)),
            },
        }

    def frame(step: int) -> dict:
        recording = state.recording
        scene = recording.scene
        step = int(np.clip(step, 0, recording.n_steps - 1))
        payload = {
            "type": "frame",
            "step": step,
            "t_ms": step * recording.dt_ms,
            "activity": _quantise(recording.activity[step], state.display_scale),
            "aggregate": {
                name: float(values[step]) for name, values in state.aggregates.items()
            },
        }
        if scene is not None:
            payload.update(
                theta_deg=float(scene.theta_deg[step]),
                theta_dot=float(scene.theta_dot_deg_per_ms[step]),
                distance_mm=float(scene.distance_mm[step]),
            )
        return payload

    @app.get("/health")
    async def health() -> dict:
        return {"ok": True, "busy": state.busy, "n_steps": state.recording.n_steps}

    @app.websocket("/stream")
    async def stream(socket: WebSocket) -> None:
        await socket.accept()
        await socket.send_text(json.dumps(hello()))

        async def receive() -> None:
            """Client commands. Runs alongside the frame pump."""
            while True:
                try:
                    message = json.loads(await socket.receive_text())
                except (WebSocketDisconnect, RuntimeError):
                    return
                except json.JSONDecodeError:
                    continue
                await handle(socket, message)

        async def pump() -> None:
            interval = 1.0 / RENDER_HZ
            while True:
                await asyncio.sleep(interval)
                playback = state.playback
                if state.busy:
                    continue
                try:
                    await socket.send_text(json.dumps(frame(playback.step)))
                except (WebSocketDisconnect, RuntimeError):
                    return
                if not playback.playing:
                    continue
                # Advance by the dilated amount of simulation time.
                advance = interval * 1000.0 * playback.speed / state.recording.dt_ms
                playback.step += max(1, int(round(advance)))
                if playback.step >= state.recording.n_steps:
                    playback.step = 0 if playback.loop else state.recording.n_steps - 1
                    playback.playing = playback.loop

        receiver = asyncio.create_task(receive())
        pumper = asyncio.create_task(pump())
        done, pending = await asyncio.wait(
            {receiver, pumper}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()

    async def handle(socket: WebSocket, message: dict) -> None:
        playback = state.playback
        command = message.get("cmd")

        if command == "play":
            playback.playing = True
        elif command == "pause":
            playback.playing = False
        elif command == "seek":
            playback.step = int(np.clip(
                int(message.get("step", 0)), 0, state.recording.n_steps - 1
            ))
        elif command == "speed":
            playback.speed = float(np.clip(float(message.get("value", DEFAULT_SPEED)),
                                           0.0005, 1.0))
        elif command == "rerun" and state.rerun is not None:
            await run_new_trial(socket, message)

    async def run_new_trial(socket: WebSocket, message: dict) -> None:
        """Simulate a fresh trial without blocking the event loop."""
        state.busy = "simulating"
        await socket.send_text(json.dumps({"type": "status", "busy": "simulating"}))
        try:
            recording, _ = await asyncio.get_running_loop().run_in_executor(
                None,
                state.rerun,
                float(message.get("ratio_ms", 40.0)),
                float(message.get("gain_scale", 0.03)),
                int(message.get("seed", 0)),
            )
            state.recording = recording
            state.aggregates = recording.aggregate_by_type()
            state.display_scale = display_scale_for(recording.activity)
            # The physics belongs to this trial, so it has to move with it.
            state.world = recording.meta.get("world")
            state.playback.step = 0
            state.playback.playing = True
        finally:
            state.busy = None
        await socket.send_text(json.dumps(hello()))

    return app
