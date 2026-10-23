#!/usr/bin/env python3
"""Small NCCL preflight for the Qwen3.5 Dense multi-node job.

The training entry launches this file with torchrun before a real job when
NCCL_PREFLIGHT_ONLY=true.  It deliberately avoids importing Megatron or the
model so a failed network/topology check is cheap and easy to diagnose.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import socket
import sys


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer, got {value!r}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-world-size",
        type=int,
        default=None,
        help="Fail if torchrun reports a different global process count.",
    )
    parser.add_argument(
        "--expected-local-world-size",
        type=int,
        default=None,
        help="Fail if torchrun reports a different local process count.",
    )
    parser.add_argument(
        "--tensor-mib",
        type=int,
        default=16,
        help="GPU tensor size for the second all-reduce (default: 16 MiB).",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=180,
        help="Process-group initialization and collective timeout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.tensor_mib < 1:
        raise SystemExit("--tensor-mib must be positive")
    if args.timeout_sec < 10:
        raise SystemExit("--timeout-sec must be at least 10 seconds")

    try:
        import torch
        import torch.distributed as dist
    except Exception as exc:  # pragma: no cover - exercised on remote nodes
        print(f"[NCCL_PREFLIGHT] unable to import torch: {exc}", file=sys.stderr)
        return 2

    rank = _int_env("RANK", 0)
    world_size = _int_env("WORLD_SIZE", 1)
    local_rank = _int_env("LOCAL_RANK", 0)
    local_world_size = _int_env("LOCAL_WORLD_SIZE", 1)
    if args.expected_world_size is not None and world_size != args.expected_world_size:
        print(
            f"[NCCL_PREFLIGHT] expected WORLD_SIZE={args.expected_world_size}, "
            f"got {world_size}",
            file=sys.stderr,
        )
        return 2
    if (
        args.expected_local_world_size is not None
        and local_world_size != args.expected_local_world_size
    ):
        print(
            f"[NCCL_PREFLIGHT] expected LOCAL_WORLD_SIZE={args.expected_local_world_size}, "
            f"got {local_world_size}",
            file=sys.stderr,
        )
        return 2
    if not torch.cuda.is_available():
        print("[NCCL_PREFLIGHT] CUDA is unavailable", file=sys.stderr)
        return 2
    device_count = torch.cuda.device_count()
    if local_rank < 0 or local_rank >= device_count:
        print(
            f"[NCCL_PREFLIGHT] LOCAL_RANK={local_rank} is outside "
            f"the {device_count} visible CUDA devices",
            file=sys.stderr,
        )
        return 2

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    hostname = socket.gethostname()
    device_name = torch.cuda.get_device_name(local_rank)
    timeout = dt.timedelta(seconds=args.timeout_sec)

    initialized = False
    try:
        dist.init_process_group(backend="nccl", init_method="env://", timeout=timeout)
        initialized = True
        actual_world_size = dist.get_world_size()
        actual_rank = dist.get_rank()
        if actual_world_size != world_size or actual_rank != rank:
            raise RuntimeError(
                "torchrun environment disagrees with the process group: "
                f"env rank/world={rank}/{world_size}, "
                f"group rank/world={actual_rank}/{actual_world_size}"
            )

        # The first reduction catches rank/device assignment and validates the
        # result without allocating a large buffer.
        value = torch.tensor(float(rank + 1), dtype=torch.float64, device=device)
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        expected_sum = world_size * (world_size + 1) / 2
        if value.item() != expected_sum:
            raise RuntimeError(
                f"rank sum mismatch: got {value.item()}, expected {expected_sum}"
            )

        # A modest payload catches NCCL transport problems that a scalar can
        # hide while staying well below the model's memory footprint.
        element_count = args.tensor_mib * 1024 * 1024 // torch.tensor([], dtype=torch.bfloat16).element_size()
        payload = torch.ones(element_count, dtype=torch.bfloat16, device=device)
        dist.all_reduce(payload, op=dist.ReduceOp.SUM)
        expected_value = float(world_size)
        if not torch.all(payload == expected_value).item():
            raise RuntimeError(
                f"payload all-reduce mismatch: expected every element to equal {expected_value}"
            )
        del payload

        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(
            gathered,
            {
                "rank": rank,
                "host": hostname,
                "device": device_name,
                "device_count": device_count,
                "capability": ".".join(map(str, torch.cuda.get_device_capability(local_rank))),
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "nccl": str(torch.cuda.nccl.version()),
            },
        )
        dist.barrier()
        if rank == 0:
            hosts = {}
            for item in gathered:
                hosts.setdefault(item["host"], []).append(item["rank"])
            if world_size % local_world_size != 0:
                raise RuntimeError(
                    f"WORLD_SIZE={world_size} is not divisible by "
                    f"LOCAL_WORLD_SIZE={local_world_size}"
                )
            expected_host_count = world_size // local_world_size
            if len(hosts) != expected_host_count:
                raise RuntimeError(
                    f"expected {expected_host_count} hosts, gathered {len(hosts)}: "
                    f"{sorted(hosts)}"
                )
            bad_device_counts = sorted(
                (item["rank"], item["device_count"])
                for item in gathered
                if item["device_count"] != local_world_size
            )
            if bad_device_counts:
                raise RuntimeError(
                    f"visible CUDA device count mismatch by rank: {bad_device_counts}"
                )
            # Megatron's default rank order forms TP groups from contiguous
            # ranks. Require each local block to stay on one host so a node-local TP group does
            # not accidentally span nodes due to a platform rank remap.
            for start in range(0, world_size, local_world_size):
                block = gathered[start : start + local_world_size]
                block_hosts = {item["host"] for item in block}
                if len(block_hosts) != 1:
                    raise RuntimeError(
                        "TP-locality check failed for ranks "
                        f"{start}..{start + local_world_size - 1}: "
                        f"hosts={sorted(block_hosts)}"
                    )
            software = {
                (item["torch"], item["cuda"], item["nccl"], item["device"], item["capability"])
                for item in gathered
            }
            if len(software) != 1:
                raise RuntimeError(f"software/device mismatch across ranks: {sorted(software)}")
            torch_version, cuda_version, nccl_version, _, capability = software.pop()
            print(
                "[NCCL_PREFLIGHT] PASS "
                f"world_size={world_size} local_world_size={local_world_size} "
                f"hosts={len(hosts)} devices={device_name}"
            )
            print(
                "[NCCL_PREFLIGHT] versions "
                f"torch={torch_version} cuda={cuda_version} nccl={nccl_version} "
                f"capability={capability}"
            )
            for host, ranks in sorted(hosts.items()):
                print(f"[NCCL_PREFLIGHT] host={host} ranks={','.join(map(str, sorted(ranks)))}")
        return 0
    except Exception as exc:  # pragma: no cover - depends on remote fabric
        print(
            f"[NCCL_PREFLIGHT] FAIL rank={rank} host={hostname}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    finally:
        if initialized:
            try:
                dist.destroy_process_group()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
