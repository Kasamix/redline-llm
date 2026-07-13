"""Benchmark harness for the redline engine.

Drives multiple inference engines behind one adapter interface
(``bench.adapters``) on identical fixed-seed token-ID workloads
(``bench.workload``), computes metrics from raw per-token timestamps
(``bench.run_bench``), and renders result JSON into markdown tables
(``bench.report``).

Methodology and apples-to-apples rules live in bench/FAIRNESS.md. Numbers are
published only after this harness has measured them on documented hardware.
"""
