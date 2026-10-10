"""Hip stream-event benchmarking."""

from klineage.backend.timing import TimingPolicy
from klineage.backend.timing import check_event_timing as check_timing

NAME = "hip-events"
DEFAULT_POLICY = TimingPolicy(cold_l2=False)


def bench(fn, *, args=(), policy=None):
    from klineage.backend.runtime import bench_events

    from .kernel import HipBackend

    return bench_events(
        fn, backend=HipBackend(), args=args, policy=policy or DEFAULT_POLICY
    )


__all__ = ["NAME", "DEFAULT_POLICY", "bench", "check_timing"]
