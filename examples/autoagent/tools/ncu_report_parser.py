#!/usr/bin/env python3
"""
NCU Report Parser for extracting optimization suggestions and performance metrics.
"""

import re
import yaml


class NCUReportParser:
    """Parse NCU text report to extract OPT blocks and key metrics."""

    def __init__(self, ncu_report="ncu_reports/log"):
        try:
            with open(ncu_report, "r") as f:
                self.text = f.read()
        except FileNotFoundError:
            self.text = ""
            print(f"Warning: NCU report {ncu_report} not found.")

        self.opt_blocks = self._extract_opt_blocks()
        self.metrics = self._extract_metrics()

    def _extract_opt_blocks(self):
        if not self.text:
            return []
        pattern = r'^ {4}OPT\s+Est.*?(?=^ {4}-)'
        return re.findall(pattern, self.text, re.S | re.M)

    def _extract_metrics(self):
        metrics = {}
        if not self.text:
            return metrics
        patterns = {
            "duration_us": r'Duration\s+us\s+([\d.]+)',
            "memory_throughput_pct": r'Memory Throughput\s+%\s+([\d.]+)',
            "compute_throughput_pct": r'Compute \(SM\) Throughput\s+%\s+([\d.]+)',
            "shared_memory_throughput_pct": r'L1/TEX Cache Throughput\s+%\s+([\d.]+)',
            "dram_throughput_pct": r'DRAM Throughput\s+%\s+([\d.]+)',
            "l2_throughput_pct": r'L2 Cache Throughput\s+%\s+([\d.]+)',
            "mem_pipes_busy_pct": r'Mem Pipes Busy\s+%\s+([\d.]+)',
            "achieved_occupancy_pct": r'Achieved Occupancy\s+%\s+([\d.]+)',
            "sm_busy_pct": r'SM Busy\s+%\s+([\d.]+)',
            "memory_throughput_gbps": r'Memory Throughput\s+Gbyte/s\s+([\d.]+)',
        }
        for key, pat in patterns.items():
            m = re.search(pat, self.text)
            if m:
                metrics[key] = float(m.group(1))
        return metrics

    def opt_text(self):
        """Return all OPT blocks as a single string."""
        return "\n\n".join(b.rstrip() for b in self.opt_blocks)

    def summary(self):
        """One-line summary of each OPT block."""
        lines = []
        for i, b in enumerate(self.opt_blocks):
            first = b.strip().split('\n')[0].strip()
            lines.append(f"[{i}] {first}")
        return "\n".join(lines)

    def print_report(self):
        for i, b in enumerate(self.opt_blocks):
            print(f"=== BLOCK {i} ===")
            print(b.rstrip())
            print()

        print("--- Key Metrics ---")
        for k, v in self.metrics.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    import sys
    log_path = sys.argv[1] if len(sys.argv) > 1 else "ncu_reports/log"
    parser = NCUReportParser(log_path)
    parser.print_report()
