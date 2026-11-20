"""
eval/gpu_state.py: a results file says whether its timings were taken while another
process was computing on the GPU (round 12 measured that as the one condition that skews
them), and reading the GPU can never fail a benchmark.
"""

import pytest

from eval import gpu_state

LINE = "NVIDIA GeForce RTX 5070 Ti Laptop GPU, 255, 12227, 0"


def _query(*utilisations):
    readings = iter(utilisations)

    def query(fields):
        return f"NVIDIA GeForce RTX 5070 Ti Laptop GPU, 6087, 12227, {next(readings)}"
    return query


def test_a_line_is_read_field_by_field_even_with_commas_in_the_name():
    assert gpu_state.parse_gpu_line(LINE) == {
        "name": "NVIDIA GeForce RTX 5070 Ti Laptop GPU",
        "memory_used_mb": 255.0, "memory_total_mb": 12227.0, "utilization_pct": 0.0,
    }
    assert gpu_state.parse_gpu_line("Tesla T4, Rev A, 10, 15360, 87")["name"] == "Tesla T4, Rev A"


def test_a_sample_keeps_the_mean_and_peak_utilisation(monkeypatch):
    monkeypatch.setattr(gpu_state.time, "sleep", lambda s: None)
    clock = iter(range(100))
    monkeypatch.setattr(gpu_state.time, "monotonic", lambda: next(clock))
    snap = gpu_state.sample(seconds=3, interval=0.5, query=_query(40, 100, 70))  # clock ticks 1 a read
    assert snap["available"] and snap["samples"] == 3
    assert snap["utilization_pct_mean"] == 70.0 and snap["utilization_pct_max"] == 100.0
    assert gpu_state.busy(snap)


def test_memory_held_by_an_idle_process_is_not_busy(monkeypatch):
    # An idle model holding most of the GPU's memory did not move any timing (round 12).
    monkeypatch.setattr(gpu_state.time, "sleep", lambda s: None)
    snap = gpu_state.sample(seconds=0, query=_query(2))
    assert snap["memory_used_mb"] == 6087.0 and not gpu_state.busy(snap)


@pytest.mark.parametrize("query", [
    lambda fields: None,                                   # no nvidia-smi / no NVIDIA GPU
    lambda fields: (_ for _ in ()).throw(OSError("boom")),  # it fails to run
    lambda fields: "garbage",                               # it prints something unexpected
])
def test_an_unreadable_gpu_is_reported_unavailable_never_raised(query):
    assert gpu_state.sample(seconds=0, query=query) == {"available": False}
    assert not gpu_state.busy({"available": False})


def test_the_summary_warns_only_when_the_gpu_was_busy_before_or_after():
    idle = {"available": True, "utilization_pct_mean": 1.0, "utilization_pct_max": 4.0}
    hot = {"available": True, "utilization_pct_mean": 62.0, "utilization_pct_max": 100.0}
    assert gpu_state.note(gpu_state.record(idle, idle)) == ""
    assert gpu_state.note(gpu_state.record({"available": False}, {"available": False})) == ""
    assert gpu_state.note(None) == ""
    line = gpu_state.note(gpu_state.record(idle, hot))
    assert line.startswith("**Timings not comparable:**") and "62% after this run (peak 100%)" in line
    assert "before" not in line


def test_the_verification_summary_carries_the_warning():
    from eval.run_verification_eval import render_markdown

    hot = {"available": True, "utilization_pct_mean": 55.0, "utilization_pct_max": 90.0}
    report = {"generated_at": "2026-11-20T12:00:00+00:00", "command": "python -m eval.run_verification_eval",
              "nli": "off", "device": "n/a", "total_seconds": 1.0, "splits": {},
              "gpu": gpu_state.record(hot, {"available": False})}
    assert "**Timings not comparable:**" in render_markdown(report)
    report["gpu"] = gpu_state.record({"available": False}, {"available": False})
    assert "Timings not comparable" not in render_markdown(report)


@pytest.mark.parametrize("device", ["cuda", "cuda (NVIDIA GeForce RTX 5070 Ti Laptop GPU)"])
def test_both_harnesses_device_strings_count_as_cuda(device, monkeypatch):
    # The golden harness reports "cuda (<name>)", the verification harness "cuda"; an
    # equality test on "cuda" silently skipped the golden run's sample.
    monkeypatch.setattr(gpu_state.time, "sleep", lambda s: None)
    assert gpu_state.sample_if(device, seconds=0, query=_query(0))["available"]


@pytest.mark.parametrize("device", ["cpu", "n/a", ""])
def test_no_gpu_run_reads_no_gpu(device):
    def query(fields):
        raise AssertionError("nvidia-smi must not be called")
    assert gpu_state.sample_if(device, seconds=0, query=query) == {"available": False}


def test_the_closing_sample_waits_out_the_runs_own_last_kernels(monkeypatch):
    slept = []
    monkeypatch.setattr(gpu_state.time, "sleep", slept.append)
    snap = gpu_state.sample_after({"available": True}, settle=2.0, seconds=0, query=_query(0))
    assert snap["available"] and slept[0] == 2.0


def test_no_closing_sample_when_the_opening_one_could_not_read_the_gpu():
    def query(fields):
        raise AssertionError("nvidia-smi must not be called")
    assert gpu_state.sample_after({"available": False}, query=query) == {"available": False}
    assert gpu_state.sample_after(None, query=query) == {"available": False}
