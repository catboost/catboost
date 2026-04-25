#!/usr/bin/env python3
"""Quick check of L4 Phase 1 dump data."""
import numpy as np
from pathlib import Path

L3 = Path("docs/sprint33/l3-iter2/data")
L4 = Path("docs/sprint33/l4-fix/data")

# S1 gradient check
g_l3 = np.fromfile(str(L3 / "mlx_grad_iter2.bin"), dtype=np.float32)
g_l3_cpu = np.fromfile(str(L3 / "cpu_grad_iter2.bin"), dtype=np.float32)

g_l4_path = L4 / "mlx_grad_iter2.bin"
if g_l4_path.exists():
    g_l4 = np.fromfile(str(g_l4_path), dtype=np.float32)
    print(f"L3 mlx grad:  sum={g_l3.sum():.8f}  mean={g_l3.mean():.8e}  absmax={np.abs(g_l3).max():.4f}")
    print(f"L4 mlx grad:  sum={g_l4.sum():.8f}  mean={g_l4.mean():.8e}  absmax={np.abs(g_l4).max():.4f}")
    print(f"CPU grad:     sum={g_l3_cpu.sum():.8f}  mean={g_l3_cpu.mean():.8e}  absmax={np.abs(g_l3_cpu).max():.4f}")
    print(f"L3 vs CPU:    max_abs_diff={np.abs(g_l3 - g_l3_cpu).max():.3e}")
    print(f"L4 vs CPU:    max_abs_diff={np.abs(g_l4 - g_l3_cpu).max():.3e}")
else:
    print("L4 grad not found")
    print(f"L3 mlx grad:  sum={g_l3.sum():.8f}")
    print(f"CPU grad:     sum={g_l3_cpu.sum():.8f}")

print()

# Histogram check
h_l3 = np.fromfile(str(L3 / "mlx_hist_d0_iter2.bin"), dtype=np.float32)
h_l4 = np.fromfile(str(L4 / "mlx_hist_d0_iter2.bin"), dtype=np.float32)
n = len(h_l3) // 2
print(f"L3 hist grad block sum: {h_l3[:n].sum():.6f}")
print(f"L4 hist grad block sum: {h_l4[:n].sum():.6f}")
print(f"L3 hist hess block sum: {h_l3[n:].sum():.1f}")
print(f"L4 hist hess block sum: {h_l4[n:].sum():.1f}")
print()

# Check top bins
print("First 10 grad bins (L3):", h_l3[:10])
print("First 10 grad bins (L4):", h_l4[:10])
