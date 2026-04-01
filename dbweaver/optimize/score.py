import math
from typing import Dict, List, Tuple
import numpy as np

def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def _robust_quantile_bounds(values: List[float], lo_q: float = 0.10, hi_q: float = 0.90) -> Tuple[float, float]:
    """
    Return robust [lo, hi] bounds using quantiles, to avoid outliers dominating the scale.
    """
    if not values:
        # Default span in log space: roughly corresponds to ~0.82x .. ~1.22x
        return (-0.2, 0.2)
    vs = sorted(values)
    n = len(vs)
    lo_i = int(lo_q * (n - 1))
    hi_i = int(hi_q * (n - 1))
    lo = vs[lo_i]
    hi = vs[hi_i]
    if hi - lo < 1e-9:
        # If distribution is too narrow, enforce a minimum span
        lo -= 0.1
        hi += 0.1
    return lo, hi

def _normalize_to_0_1(x: float, history: List[float]) -> float:
    lo, hi = _robust_quantile_bounds(history, 0.10, 0.90)
    y = (x - lo) / (hi - lo)
    return _clamp(y, 0.0, 1.0)


def create_reflection_from_candidate(state, candidate, root_processing_time, best_candidate_processing_time):
    from state import Reflection

    candidate["parent_processing_time"] = best_candidate_processing_time

    if candidate.get("success") and candidate.get("optimized_code"):
        perf = candidate.get("performance") or {}
        processing_time = float(perf.get("processing_time", 0.0) or 0.0)
        total_time = float(perf.get("total_time", 0.0) or 0.0)


        speedup_root = (root_processing_time - processing_time) / root_processing_time

        if speedup_root < -1:
            speedup_root = -1
        if speedup_root > 1:
            speedup_root = 1

        score = 50+speedup_root*100/2.0


        if best_candidate_processing_time > 0 and processing_time > 0:
            speedup_parent = best_candidate_processing_time / processing_time
        else:
            speedup_parent = 0.0

        found_solution = (speedup_root >= 0.9)
        score = int(score) if score > 1 else 1

        reflection = Reflection(
            reflections=(
                f"Speedup(vs sketch): {speedup_root:.4f}x | "
                f"Speedup(vs parent): {speedup_parent:.6f}x | "
                f"Score: {score}/100 | "
                f"Processing: {processing_time:.6f}s (baseline: {root_processing_time:.6f}s) | "
                f"Total: {total_time:.6f}s"
            ),
            score=score,
            found_solution=found_solution,
            new_cpp_code=candidate["optimized_code"],
            performance=perf
        )

        print(f"[Reflection] speedup_root={speedup_root:.6f}x, speedup_parent={speedup_parent:.6f}x, score={score}/100")
        return reflection

    else:
        reflection = Reflection(
            reflections=f"Optimization failed: {candidate.get('message', '')}",
            performance=candidate.get("performance"),
            score=0,
            found_solution=False,
            new_cpp_code=None
        )
        return reflection
