"""
GPU Monte-Carlo Search for Rigged Dice (Farkle-like game)
========================================================
What this script does:
- Given an inventory of dice, it evaluates ALL possible 6-die lineups.
- Uses GPU-accelerated Monte-Carlo simulation to estimate:
    * Expected value (EV)
    * Farkle rate
    * Hot-dice runaway probability
- Lineups that reliably trigger repeated hot-dice are treated as INFINITE value.
- Dominated lineups (subsets of infinite lineups) are skipped.
Key features:
- Fully automatic: user declares inventory only.
- GPU-first design (PyTorch, CUDA).
- Safety caps to avoid infinite simulations.
- CLI presets for fast scans vs accurate evaluation.
Usage:
  python gpu_search.py # default (balanced)
  python gpu_search.py --quick # very fast, noisy
  python gpu_search.py --accurate # slow, stable
  python gpu_search.py -v # verbose logging
  python gpu_search.py --quick -v # fast + noisy logs
"""
import argparse
import torch, time, traceback
from collections import Counter
from typing import List, Dict, Tuple

# ----------------------------- Top-level configurable params -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N = 6 # dice per lineup
HOT_CAP = 3 # hot-dice resets to consider "runaway"

# Quick-scan defaults
SIMS_PER_LINEUP = 100
BATCH_SIZE = 1024
NUM_REPEATS = 2
MIN_SIMS_FOR_INFINITE = 2000
STABILITY_THRESHOLD = 0.95
LINEUP_CHUNK_SIZE = 12
SHORT_CIRCUIT_ON_INFINITE = False

# -----------------------------------------------------------------------------------------
# ----------------------------- Preset configurations ------------------------------------
PRESETS = {
    "quick": {
        "SIMS_PER_LINEUP": 3,
        "BATCH_SIZE": 256,
        "NUM_REPEATS": 1,
        "STABILITY_THRESHOLD": 0.90,
        "MIN_SIMS_FOR_INFINITE": 3,
        "SHORT_CIRCUIT_ON_INFINITE": True,
    },
    "accurate": {
        "SIMS_PER_LINEUP": 10_000,
        "BATCH_SIZE": 4096,
        "NUM_REPEATS": 20,
        "STABILITY_THRESHOLD": 0.985,
        "MIN_SIMS_FOR_INFINITE": 2000,
        "SHORT_CIRCUIT_ON_INFINITE": False,
    }
}

# ----------------------------- Inventory configuration ------------------------------------
# <<< EDIT YOUR INVENTORY HERE >>>
INVENTORY: Dict[str, int] = {
    "Saint Antiochus' die": 6,
}
# -----------------------------------------------------------------------------------------

# ----------------------------- Dice probabilities (paste your full list) -----------------
ALL_DICE = {
    "Aranka's die": [0.286, 0.048, 0.286, 0.048, 0.286, 0.048],
    "Cautious cheater's die": [0.238, 0.143, 0.095, 0.143, 0.238, 0.143],
    "Ci die": [0.13, 0.13, 0.13, 0.13, 0.13, 0.348],
    "Devil's head die": [0.167]*6,
    "Die of misfortune": [0.045, 0.227, 0.227, 0.227, 0.227, 0.045],
    "Even die": [0.067, 0.267, 0.067, 0.267, 0.067, 0.267],
    "Favourable die": [0.333, 0.0, 0.056, 0.056, 0.333, 0.222],
    "Fer die": [0.13, 0.13, 0.13, 0.13, 0.13, 0.348],
    "Greasy die": [0.176, 0.118, 0.176, 0.118, 0.176, 0.235],
    "Grimy die": [0.063, 0.313, 0.063, 0.063, 0.438, 0.063],
    "Grozav's lucky die": [0.067, 0.667, 0.067, 0.067, 0.067, 0.067],
    "Heavenly Kingdom die": [0.368, 0.105, 0.105, 0.105, 0.105, 0.211],
    "Holy Trinity die": [0.182, 0.227, 0.455, 0.045, 0.045, 0.045],
    "Hugo's Die": [0.167]*6,
    "King's die": [0.125, 0.188, 0.219, 0.25, 0.125, 0.094],
    "Lousy gambler's die": [0.1, 0.15, 0.1, 0.15, 0.35, 0.15],
    "Lu die": [0.13, 0.13, 0.13, 0.13, 0.13, 0.348],
    "Lucky Die": [0.273, 0.045, 0.091, 0.136, 0.182, 0.273],
    "Mathematician's Die": [0.167, 0.208, 0.25, 0.292, 0.042, 0.042],
    "Molar die": [0.167]*6,
    "Odd die": [0.267, 0.067, 0.267, 0.067, 0.267, 0.067],
    "Ordinary die": [1/6.0]*6,
    "Painted die": [0.188, 0.063, 0.063, 0.063, 0.438, 0.188],
    "Pie die": [0.462, 0.077, 0.231, 0.231, 0.0, 0.0],
    "Premolar die": [0.167]*6,
    "Sad Greaser's Die": [0.261, 0.261, 0.043, 0.043, 0.261, 0.13],
    "Saint Antiochus' die": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "Shrinking die": [0.222, 0.111, 0.111, 0.111, 0.111, 0.333],
    "St. Stephen's die": [0.167]*6,
    "Strip die": [0.25, 0.125, 0.125, 0.125, 0.188, 0.188],
    "Three die": [0.125, 0.063, 0.563, 0.063, 0.125, 0.063],
    "Unbalanced Die": [0.25, 0.333, 0.083, 0.083, 0.167, 0.083],
    "Unlucky die": [0.091, 0.273, 0.182, 0.182, 0.182, 0.091],
    "Wagoner's die": [0.056, 0.278, 0.333, 0.111, 0.111, 0.111],
    "Weighted die": [0.667, 0.067, 0.067, 0.067, 0.067, 0.067],
    "Wisdom tooth die": [0.167]*6,
}
# -----------------------------------------------------------------------------------------
# ----------------------------- scoring lookup precompute ---------------------------------
SCORING_TABLE = {
    "1": 100, "5": 50,
    "1-2-3-4-5": 500, "2-3-4-5-6": 750, "1-2-3-4-5-6": 1500,
    "1-1-1": 1000, "2-2-2": 200, "3-3-3": 300, "4-4-4": 400, "5-5-5": 500, "6-6-6": 600,
    "1-1-1-1": 2000, "2-2-2-2": 400, "3-3-3-3": 600, "4-4-4-4": 800, "5-5-5-5": 1000, "6-6-6-6": 1200,
    "1-1-1-1-1": 4000, "2-2-2-2-2": 800, "3-3-3-3-3": 1200, "4-4-4-4-4": 1600, "5-5-5-5-5": 2000, "6-6-6-6-6": 2400,
    "1-1-1-1-1-1": 8000, "2-2-2-2-2-2": 1600, "3-3-3-3-3-3": 2400, "4-4-4-4-4-4": 3200, "5-5-5-5-5-5": 4000, "6-6-6-6-6-6": 4800,
}
def build_face_score_lookup(device):
    lookup = torch.zeros((6, 7), dtype=torch.int32)
    for face in range(1, 7):
        for k in range(0, 7):
            if k == 0:
                s = 0
            elif k == 1:
                s = 100 if face == 1 else (50 if face == 5 else 0)
            elif k == 2:
                s = 200 if face == 1 else (100 if face == 5 else 0)
            else:
                key = "-".join([str(face)] * k)
                s = SCORING_TABLE.get(key, 0)
            lookup[face-1, k] = int(s)
    return lookup.to(DEVICE)
FACE_SCORE_LOOKUP = build_face_score_lookup(DEVICE)
# Masks precompute
MASKS = [m for m in range(1, 1 << N)]
M = len(MASKS) # 63
MASK_BITS = torch.tensor([[((mask >> i) & 1) for i in range(N)] for mask in MASKS],
                         dtype=torch.bool, device=DEVICE)
MASK_SIZES = MASK_BITS.sum(dim=1).to(torch.int64)
ORDER_BY_SIZE = torch.argsort(MASK_SIZES)
INDICES_BY_SIZE = {d: (MASK_SIZES == d).nonzero(as_tuple=True)[0] for d in range(1, N+1)}
# ----------------------------- utils: inventory & combinations ---------------------------
def normalize_inventory(inventory: Dict[str,int]) -> Dict[str,int]:
    normalized = {}
    ord_prob = ALL_DICE.get("Ordinary die")
    for die, cnt in inventory.items():
        if die not in ALL_DICE:
            raise KeyError(f"Die '{die}' not found in ALL_DICE.")
        if ALL_DICE[die] == ord_prob:
            normalized["Ordinary die"] = normalized.get("Ordinary die", 0) + cnt
        else:
            normalized[die] = normalized.get(die, 0) + cnt
    return normalized
def generate_rigged_combinations(inventory: Dict[str,int], max_count: int = N) -> List[List[str]]:
    items = list(inventory.items())
    combos = []
    def helper(i, cur, remaining):
        if i == len(items):
            combos.append(list(cur))
            return
        die, avail = items[i]
        for take in range(min(avail, remaining) + 1):
            cur.extend([die]*take)
            helper(i+1, cur, remaining - take)
            for _ in range(take):
                cur.pop()
    helper(0, [], max_count)
    return combos
# ----------------------------- sampling and scoring (vectorized) -------------------------
def sample_faces_for_lineup(lineup: List[str], B: int):
    probs = torch.stack([torch.tensor(ALL_DICE[die], dtype=torch.float32, device=DEVICE) for die in lineup], dim=0) # (N,6)
    cum = probs.cumsum(dim=1)
    u = torch.rand((B, N), device=DEVICE)
    faces = (u.unsqueeze(-1) < cum.unsqueeze(0)).to(torch.int8).argmax(dim=2) + 1
    return faces
def compute_scores_for_rolls_given_present(rolls: torch.Tensor, present_mask: torch.Tensor) -> torch.Tensor:
    B = rolls.shape[0]
    faces = torch.arange(1,7, device=DEVICE).view(1,1,6)
    ro = (rolls.unsqueeze(-1) == faces).to(torch.int32)
    mb = MASK_BITS.view(1, M, N, 1)
    ro_exp = ro.view(B, 1, N, 6)
    selected = ro_exp * mb
    counts = selected.sum(dim=2)
    per_face_sum = torch.zeros((B, M), dtype=torch.int32, device=DEVICE)
    for f in range(6):
        cnt = counts[:, :, f]
        per_face_sum += FACE_SCORE_LOOKUP[f, cnt].to(torch.int32)
    scores = per_face_sum
    pm = (~present_mask).view(B, 1, N)
    forbidden = (MASK_BITS.view(1, M, N) & pm).any(dim=2)
    scores = torch.where(forbidden, torch.zeros_like(scores), scores)
    mask_sizes_b = MASK_SIZES.view(1, M).expand(B, -1)
    if N == 6:
        is_size6 = (mask_sizes_b == 6)
        all_faces_present = (counts.min(dim=2).values >= 1)
        is_6_straight = is_size6 & all_faces_present
        scores = torch.where(is_6_straight, torch.maximum(scores, torch.full_like(scores, 1500)), scores)
    is_size5 = (mask_sizes_b == 5)
    if is_size5.any():
        faces_1_5 = (counts[:, :, 0:5].min(dim=2).values >= 1) & (counts[:, :, 0:5].sum(dim=2) == 5)
        faces_2_6 = (counts[:, :, 1:6].min(dim=2).values >= 1) & (counts[:, :, 1:6].sum(dim=2) == 5)
        is_1_5 = is_size5 & faces_1_5
        is_2_6 = is_size5 & faces_2_6
        scores = torch.where(is_1_5, torch.maximum(scores, torch.full_like(scores, 500)), scores)
        scores = torch.where(is_2_6, torch.maximum(scores, torch.full_like(scores, 750)), scores)
    return torch.clamp(scores, min=0)
def choose_mask_scores(scores: torch.Tensor, present_mask: torch.Tensor, min_desired: int):
    B = scores.shape[0]
    device = scores.device
    order = ORDER_BY_SIZE
    chosen_score = torch.zeros(B, dtype=torch.int32, device=device)
    chosen_idx = torch.full((B,), -1, dtype=torch.int64, device=device)
    cand = scores >= min_desired
    any_cand = cand.any(dim=1)
    if any_cand.any():
        scores_ordered = scores[:, order]
        cand_ordered = cand[:, order]
        candidate_scores = torch.where(cand_ordered, scores_ordered, torch.full_like(scores_ordered, -1))
        max_vals, argmax_idx = candidate_scores.max(dim=1)
        chosen_score = torch.where(any_cand, max_vals.to(torch.int32), chosen_score)
        chosen_idx = torch.where(any_cand, order[argmax_idx], chosen_idx)
    none_mask = ~any_cand
    if none_mask.any():
        idxs = none_mask.nonzero(as_tuple=True)[0]
        sub_scores = scores[idxs, :]
        bsub = sub_scores.shape[0]
        max_by_size = []
        for d in range(1, N+1):
            mask_idx = INDICES_BY_SIZE.get(d, torch.tensor([], device=device, dtype=torch.long))
            if mask_idx.numel() == 0:
                max_by_size.append(torch.zeros(bsub, dtype=torch.int32, device=device))
            else:
                vals = sub_scores[:, mask_idx]
                maxvals = vals.max(dim=1).values.to(torch.int32)
                max_by_size.append(maxvals)
        max_by_size_t = torch.stack(max_by_size, dim=1)
        has_any = (max_by_size_t > 0)
        if has_any.any():
            first_true = has_any.to(torch.int8).argmax(dim=1)
            for local_i, global_i in enumerate(idxs.tolist()):
                d_index = int(first_true[local_i].item())
                d = d_index + 1
                mask_idx = INDICES_BY_SIZE.get(d, torch.tensor([], device=device, dtype=torch.long))
                if mask_idx.numel() == 0:
                    chosen_score[global_i] = 0
                    chosen_idx[global_i] = -1
                else:
                    vals = sub_scores[local_i, mask_idx]
                    arg_local = int(vals.argmax().item())
                    chosen_score[global_i] = int(vals[arg_local].item())
                    chosen_idx[global_i] = int(mask_idx[arg_local].item())
    return chosen_score, chosen_idx
# ----------------------------- full-turn batch sim --------------------------------------
def simulate_full_turns_for_lineup_batch(lineup: List[str], B: int, min_desired: int):
    probs = torch.stack([torch.tensor(ALL_DICE[die], dtype=torch.float32, device=DEVICE) for die in lineup], dim=0)
    cum = probs.cumsum(dim=1)
    turn_scores = torch.zeros(B, dtype=torch.int64, device=DEVICE)
    present = torch.ones((B, N), dtype=torch.bool, device=DEVICE)
    dice_left = present.sum(dim=1)
    hot_counts = torch.zeros(B, dtype=torch.int32, device=DEVICE)
    active = torch.ones(B, dtype=torch.bool, device=DEVICE)
    finished_by_farkle = torch.zeros(B, dtype=torch.bool, device=DEVICE)
    finished_by_runaway = torch.zeros(B, dtype=torch.bool, device=DEVICE)
    max_rolls_per_turn = 200
    for _iter in range(max_rolls_per_turn):
        if not active.any():
            break
        u = torch.rand((B, N), device=DEVICE)
        faces = (u.unsqueeze(-1) < cum.unsqueeze(0)).to(torch.int8).argmax(dim=2) + 1
        scores = compute_scores_for_rolls_given_present(faces, present)
        chosen_scores, chosen_idx = choose_mask_scores(scores, present, min_desired)
        newly_farkle = (chosen_idx == -1) & active
        if newly_farkle.any():
            finished_by_farkle = finished_by_farkle | newly_farkle
            active = active & (~newly_farkle)
        to_update = (chosen_idx != -1) & active
        if to_update.any():
            idxs = to_update.nonzero(as_tuple=True)[0]
            sel_scores = chosen_scores[idxs]
            turn_scores[idxs] += sel_scores.to(torch.int64)
            chosen_idx_vals = chosen_idx[idxs]
            mask_bits_chosen = MASK_BITS[chosen_idx_vals]
            updated_present = present[idxs] & (~mask_bits_chosen)
            present[idxs] = updated_present
            dice_left[idxs] = updated_present.sum(dim=1)
            zeros = (dice_left[idxs] == 0)
            if zeros.any():
                zero_local = zeros.nonzero(as_tuple=True)[0]
                zero_global = idxs[zero_local]
                present[zero_global] = True
                dice_left[zero_global] = N
                hot_counts[zero_global] += 1
                newly_runaway = (hot_counts >= HOT_CAP) & active
                if newly_runaway.any():
                    finished_by_runaway = finished_by_runaway | newly_runaway
                    active = active & (~newly_runaway)
    total_score_sum = int(turn_scores.sum().item())
    num_farkles = int(finished_by_farkle.sum().item())
    num_runaways = int(finished_by_runaway.sum().item())
    if num_runaways > B:
        num_runaways = min(num_runaways, B)
    return total_score_sum, num_farkles, num_runaways
# ----------------------------- repeated sims wrapper with optional verbose ----------------------
def simulate_lineup_stats(lineup: List[str], sims_per_lineup: int, batch_size: int, min_desired: int, num_repeats: int, verbose: bool = False):
    sims_each = max(1, sims_per_lineup // num_repeats)
    total_score = 0
    total_farkles = 0
    total_runaways = 0
    total_simulated = 0

    if verbose:
        print(f"[SIM-START] lineup={lineup} sims_per_lineup={sims_per_lineup} sims_each={sims_each} batch_size={batch_size} repeats={num_repeats} device={DEVICE}")

    for repeat_idx in range(num_repeats):
        seed = (int(time.time()) + repeat_idx) & 0xFFFFFFFF
        torch.manual_seed(seed)
        if DEVICE.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
        if verbose:
            print(f"[SIM] repeat {repeat_idx+1}/{num_repeats}, seed={seed}")

        sims_done = 0
        batch_count = 0
        PRINT_EVERY = 2

        while sims_done < sims_each:
            b = min(batch_size, sims_each - sims_done)
            batch_count += 1

            if verbose:
                print(f"[BATCH-START] lineup={lineup} repeat={repeat_idx+1} batch={batch_count} batch_size={b} sims_done={sims_done}/{sims_each}")

            t0 = time.time()
            try:
                total, nf, nr = simulate_full_turns_for_lineup_batch(lineup, b, min_desired)
            except Exception as e:
                print(f"[ERROR] simulate_full_turns_for_lineup_batch failed on lineup={lineup} batch_size={b}: {e}")
                traceback.print_exc()
                raise
            dt = time.time() - t0

            total_score += total
            total_farkles += nf
            total_runaways += nr
            sims_done += b
            total_simulated += b

            if verbose and ((batch_count % PRINT_EVERY == 0) or (sims_done == sims_each)):
                if DEVICE.type == 'cuda':
                    try:
                        mem_alloc = torch.cuda.memory_allocated() / (1024**2)
                        mem_reserved = torch.cuda.memory_reserved() / (1024**2)
                    except Exception:
                        mem_alloc = mem_reserved = 0.0
                else:
                    mem_alloc = mem_reserved = 0.0
                print(
                    f"[PROG] lineup={lineup} repeat={repeat_idx+1} batch={batch_count} "
                    f"done={sims_done}/{sims_each} batch_size={b} dt={dt:.2f}s "
                    f"tot_score={total_score} tot_farkles={total_farkles} tot_runaways={total_runaways} "
                    f"GPU_mem_alloc={mem_alloc:.1f}MB reserved={mem_reserved:.1f}MB"
                )

    if verbose:
        print(
            f"[SIM-END] lineup={lineup} summary:\n"
            f" - total_simulated = {total_simulated}\n"
            f" - total_score = {total_score}\n"
            f" - total_farkles = {total_farkles}\n"
            f" - total_runaways = {total_runaways}"
        )

    return total_simulated, total_score, total_farkles, total_runaways
# ----------------------------- dominance helpers ----------------------------------------
def rigged_counts_from_lineup(lineup: List[str]) -> Dict[str,int]:
    c = Counter()
    for d in lineup:
        if d != "Ordinary die":
            c[d] += 1
    return dict(c)
def is_subset_rigged(a_lineup: List[str], b_lineup: List[str]) -> bool:
    a = rigged_counts_from_lineup(a_lineup)
    b = rigged_counts_from_lineup(b_lineup)
    for die, cnt in a.items():
        if cnt > b.get(die, 0):
            return False
    return True
# ----------------------------- top-level search -----------------------------------------
def best_rigged_selection_gpu(inventory: Dict[str,int],
                              min_desired: int = 500,
                              sims_per_lineup: int = SIMS_PER_LINEUP,
                              batch_size: int = BATCH_SIZE,
                              num_repeats: int = NUM_REPEATS,
                              stability_threshold: float = STABILITY_THRESHOLD,
                              lineup_chunk_size: int = LINEUP_CHUNK_SIZE,
                              short_circuit_on_infinite: bool = SHORT_CIRCUIT_ON_INFINITE,
                              min_sims_for_infinite: int = MIN_SIMS_FOR_INFINITE):
    inv = normalize_inventory(inventory)
    combos = generate_rigged_combinations(inv, N)
    combos_sorted = sorted(combos, key=lambda c: (-len(c), c))
    candidates = [combo + ["Ordinary die"]*(N - len(combo)) for combo in combos_sorted]
    print(f"[SEARCH] Candidates generated: {len(candidates)}. Device={DEVICE}. Testing order prefers more rigged dice first.")
    best_lineup = None
    best_score = -1.0
    results = []
    infinite_lineups = []
    for chunk_start in range(0, len(candidates), lineup_chunk_size):
        chunk = candidates[chunk_start:chunk_start + lineup_chunk_size]
        print(f"[SEARCH] Evaluating chunk {chunk_start // lineup_chunk_size + 1} ({len(chunk)} lineups)...")
        for lineup in chunk:
            dominated = False
            for inf_lineup in infinite_lineups:
                if is_subset_rigged(lineup, inf_lineup):
                    print(f" [SKIP] {lineup} subset of infinite {inf_lineup}")
                    dominated = True
                    break
            if dominated:
                continue
            total_sims, total_score, total_farkles, total_runaways = simulate_lineup_stats(
                lineup, sims_per_lineup, batch_size, min_desired, num_repeats, verbose=False  # verbose passed from main
            )
            if total_sims == 0:
                print(f" [WARN] no sims recorded for {lineup}, skipping")
                continue
            ev = total_score / total_sims
            farkle_rate = total_farkles / total_sims
            runaway_frac = total_runaways / total_sims
            if total_sims >= min_sims_for_infinite and runaway_frac >= stability_threshold:
                hybrid_score = float("inf")
                print(f" [INFINITE] {lineup} runaway_frac={runaway_frac:.3f} (INF)")
                infinite_lineups.append(lineup)
                results.append((lineup, hybrid_score))
                best_lineup = lineup
                best_score = float("inf")
                if short_circuit_on_infinite:
                    print("[SEARCH] short-circuit on infinite -> stop")
                    return best_lineup, best_score, results
                continue
            else:
                hybrid_score = ev * (1.0 - farkle_rate)
                print(f" [NUMERIC] {lineup} EV={ev:.2f} farkle={farkle_rate:.3f} runaway={runaway_frac:.3f} hybrid={hybrid_score:.2f}")
                results.append((lineup, hybrid_score))
                if hybrid_score > best_score:
                    best_score = hybrid_score
                    best_lineup = lineup
    if infinite_lineups:
        best_inf = None
        best_frac = -1.0
        for inf in infinite_lineups:
            ts, _, _, tr = simulate_lineup_stats(inf, max(min_sims_for_infinite, sims_per_lineup//10), batch_size, min_desired, max(1, num_repeats//2), verbose=False)
            frac = tr / ts if ts > 0 else 0.0
            if frac > best_frac:
                best_frac = frac
                best_inf = inf
        return best_inf, float("inf"), results
    return best_lineup, best_score, results
# ----------------------------- Example run ----------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Monte-Carlo rigged-dice lineup search")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose per-batch debug output")
    parser.add_argument("--quick", action="store_true",
                        help="Fast, noisy preset (few sims, good for exploration)")
    parser.add_argument("--accurate", action="store_true",
                        help="Slow, stable preset (many sims, reliable results)")
    args = parser.parse_args()

    VERBOSE = bool(args.verbose)

    if args.quick and args.accurate:
        raise ValueError("Choose only one preset: --quick OR --accurate")
    if args.quick:
        preset = PRESETS["quick"]
        SIMS_PER_LINEUP = preset["SIMS_PER_LINEUP"]
        BATCH_SIZE = preset["BATCH_SIZE"]
        NUM_REPEATS = preset["NUM_REPEATS"]
        STABILITY_THRESHOLD = preset["STABILITY_THRESHOLD"]
        MIN_SIMS_FOR_INFINITE = preset["MIN_SIMS_FOR_INFINITE"]
        SHORT_CIRCUIT_ON_INFINITE = preset["SHORT_CIRCUIT_ON_INFINITE"]
        print("[PRESET] QUICK mode enabled")
    elif args.accurate:
        preset = PRESETS["accurate"]
        SIMS_PER_LINEUP = preset["SIMS_PER_LINEUP"]
        BATCH_SIZE = preset["BATCH_SIZE"]
        NUM_REPEATS = preset["NUM_REPEATS"]
        STABILITY_THRESHOLD = preset["STABILITY_THRESHOLD"]
        MIN_SIMS_FOR_INFINITE = preset["MIN_SIMS_FOR_INFINITE"]
        SHORT_CIRCUIT_ON_INFINITE = preset["SHORT_CIRCUIT_ON_INFINITE"]
        print("[PRESET] ACCURATE mode enabled")
    else:
        print("[PRESET] Default (balanced) settings")

    if VERBOSE:
        print("Verbose mode enabled: showing detailed per-batch simulation progress")

    print("Starting search...")
    best_lineup, best_score, all_results = best_rigged_selection_gpu(
        INVENTORY,
        min_desired=500,
        sims_per_lineup=SIMS_PER_LINEUP,
        batch_size=BATCH_SIZE,
        num_repeats=NUM_REPEATS,
        stability_threshold=STABILITY_THRESHOLD,
        lineup_chunk_size=LINEUP_CHUNK_SIZE,
        short_circuit_on_infinite=SHORT_CIRCUIT_ON_INFINITE,
        min_sims_for_infinite=MIN_SIMS_FOR_INFINITE
    )
    print("\n=== FINAL RESULT ===")
    if best_score == float("inf"):
        print("Best lineup is INFINITE (hot-dice runaway detected):")
    else:
        print(f"Best numeric lineup with hybrid score {best_score:.2f}:")
    print(best_lineup)
    print("\nHere are some other results:")
    for lineup, score in all_results[:20]:
        print(f" {lineup} -> {'INF' if score == float('inf') else f'{score:.2f}'}")
