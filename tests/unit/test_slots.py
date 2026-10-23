"""The in-flight limit and queue of a backend (``providers/http.py``, M6): arrival order or
earliest deadline first, the queue timeout plus the request's slack, and a slot is never lost
when a waiter gives up."""

from __future__ import annotations

import asyncio

import pytest

from mlx_realtime.providers.base import ProviderError
from mlx_realtime.providers.http import Slots


async def settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0)


async def served_order(slots: Slots, deadlines: list[float | None]) -> list[int]:
    """Hold the one slot, queue one request per deadline, release, and record the order."""
    order: list[int] = []
    release = asyncio.Event()

    async def holder() -> None:
        async with slots.slot():
            await release.wait()

    async def request(index: int, deadline: float | None) -> None:
        async with slots.slot(deadline=deadline):
            order.append(index)

    async with asyncio.TaskGroup() as tasks:
        tasks.create_task(holder())
        await settle()
        for index, deadline in enumerate(deadlines):
            tasks.create_task(request(index, deadline))
            await settle()
        assert slots.waiting == len(deadlines)
        release.set()
    assert slots.waiting == 0
    return order


async def test_fifo_serves_in_arrival_order_whatever_the_deadlines() -> None:
    slots = Slots(1, 5.0)
    assert await served_order(slots, [30.0, 10.0, 20.0]) == [0, 1, 2]


async def test_edf_serves_the_earliest_deadline_first_and_ties_in_arrival_order() -> None:
    slots = Slots(1, 5.0, policy="edf", now=lambda: 15.0)
    # A request without a deadline counts from its arrival: now, 15.
    assert await served_order(slots, [30.0, 10.0, 20.0, None, 10.0]) == [1, 4, 3, 2, 0]


async def test_a_free_slot_is_taken_at_once_even_with_a_zero_queue_timeout() -> None:
    slots = Slots(2, 0.0, policy="edf")
    async with slots.slot(deadline=1.0), slots.slot(deadline=2.0):
        assert slots.waiting == 0
        with pytest.raises(ProviderError) as raised:
            async with slots.slot(deadline=0.0):
                pass
    assert raised.value.kind == "overloaded"


async def test_the_wait_is_bounded_by_the_queue_timeout_plus_the_slack() -> None:
    slots = Slots(1, 0.05, policy="edf")
    loop = asyncio.get_running_loop()
    async with slots.slot():
        started = loop.time()
        with pytest.raises(ProviderError):
            async with slots.slot(deadline=1.0, slack_s=0.1):
                pass
        assert loop.time() - started >= 0.14
    assert slots.waiting == 0
    async with slots.slot():  # the slot came back
        pass


async def test_a_cancelled_waiter_passes_on_a_slot_it_was_just_handed() -> None:
    slots = Slots(1, 5.0, policy="edf")
    release = asyncio.Event()
    got: list[str] = []

    async def holder() -> None:
        async with slots.slot():
            await release.wait()

    async def request(name: str, deadline: float) -> None:
        async with slots.slot(deadline=deadline):
            got.append(name)

    hold = asyncio.create_task(holder())
    await settle()
    first = asyncio.create_task(request("first", 1.0))
    second = asyncio.create_task(request("second", 2.0))
    await settle()
    release.set()
    await asyncio.sleep(0)  # the holder hands the slot to "first"...
    first.cancel()  # ...which is cancelled before it runs
    await asyncio.gather(hold, first, second, return_exceptions=True)
    assert got == ["second"]
    assert slots.waiting == 0
    async with asyncio.timeout(1), slots.slot():
        pass
