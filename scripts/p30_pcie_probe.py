"""P30: MUSA host<->device copy bandwidth probe (read-only, informational).

Pre-registered question (docs/P30-OPPORTUNITIES.md section 4.5): is pinned-
memory D2H/H2D bandwidth on this box sufficient to make a SGLang-HiCache-style
KV offload (GPU->CPU swap of LRU-evicted blocks) plausible?

Gate (informational, decided BEFORE running):
  D2H >= 5 GB/s  -> one TP2 30B KV block (6 MB) swaps in ~1.2 ms, hideable
                    next to a 142 ms decode step ->立项候选.
  D2H < 2 GB/s   -> swap cost swamps the scheduling budget -> NO-GO 照录.
  in between     -> record numbers; re-judge only on real-traffic demand.

Sizes: 6 MB and 64 MB. NOTE (P30-R1 correction): one TP2 30B KV block is
12 MiB (256 tok x 48 KiB/token/rank: 2 x 48L x 2 kvh x 128 x bf16), not
6 MB as first written here — 6 MB remains a valid small-transfer point
and the 64 MB point covers the bulk regime; the gate math below reads
the same either way at the recorded bandwidth.
>=5 timed reps, median + CV, warmup rep discarded.
No engine code involved; one idle GPU; finishes in well under a minute.
"""
import argparse
import json
import statistics
import time

import torch
import torch_musa  # noqa: F401  (registers the musa backend)


def bench(size_bytes, reps, pin):
    host = torch.empty(size_bytes, dtype=torch.uint8, pin_memory=pin)
    dev = torch.empty(size_bytes, dtype=torch.uint8, device="musa")
    out = {}
    # d2h reads the device into host memory; h2d writes the device from host.
    # (Review fix: the first version had these two lambdas swapped — both
    # directions were measured with the same method and came out near-equal,
    # so the recorded 29.68/29.80 pair stands; only the labels were wrong.)
    for name, op in (("d2h", lambda: host.copy_(dev, non_blocking=True)),
                     ("h2d", lambda: dev.copy_(host, non_blocking=True))):
        op()
        torch.musa.synchronize()          # warmup + I28: sync before timing D2H
        samples = []
        for _ in range(reps):
            t0 = time.perf_counter()
            op()
            torch.musa.synchronize()
            samples.append(size_bytes / (time.perf_counter() - t0) / 1e9)
        out[name] = {
            "median_gbps": round(statistics.median(samples), 2),
            "cv": round(statistics.pstdev(samples) / statistics.mean(samples), 4),
            "samples": [round(s, 2) for s in samples],
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--output", default="/root/p30_data/pcie_probe.json")
    args = parser.parse_args()
    payload = {"schema_version": 1, "reps": args.reps, "sizes": {}}
    for mb, pin in ((6, True), (64, True), (6, False)):
        payload["sizes"][f"{mb}MB_pin{int(pin)}"] = bench(mb * 1024 * 1024,
                                                          args.reps, pin)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    print(json.dumps(payload, indent=2))
    d2h = payload["sizes"]["6MB_pin1"]["d2h"]["median_gbps"]
    verdict = ("PLAUSIBLE (>=5 GB/s)" if d2h >= 5
               else "NO-GO SIGNAL (<2 GB/s)" if d2h < 2
               else "GRAY ZONE (2-5 GB/s)")
    print(f"[p30-pcie-probe] 6MB pinned D2H median {d2h} GB/s -> {verdict}")


if __name__ == "__main__":
    main()
