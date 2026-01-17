"""
KCD2 Rigged Dice Simulator (Optimized)
---------------------------------------
This script evaluates various candidate combinations of rigged dice to determine the optimal 6-dice lineup for a game turn.
It now implements several optimizations, including caching of outcome data across states, refined state space handling,
and multiprocessing improvements.
"""

import itertools, math, sys, os
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed

# Increase recursion limit for the partitioning functions
sys.setrecursionlimit(10000)

##############################################
# 1. Dice and Scoring Definitions
##############################################

dice_probabilities = {
    "Aranka's die":            [0.286, 0.048, 0.286, 0.048, 0.286, 0.048],
    "Cautious cheater's die":   [0.238, 0.143, 0.095, 0.143, 0.238, 0.143],
    "Ci die":                   [0.13,  0.13,  0.13,  0.13,  0.13,  0.348],
    "Devil's head die":         [0.167]*6,
    "Die of misfortune":        [0.045, 0.227, 0.227, 0.227, 0.227, 0.045],
    "Even die":                 [0.067, 0.267, 0.067, 0.267, 0.067, 0.267],
    "Favourable die":           [0.333, 0.0,   0.056, 0.056, 0.333, 0.222],
    "Fer die":                  [0.13,  0.13,  0.13,  0.13,  0.13,  0.348],
    "Greasy die":               [0.176, 0.118, 0.176, 0.118, 0.176, 0.235],
    "Grimy die":                [0.063, 0.313, 0.063, 0.063, 0.438, 0.063],
    "Grozav's lucky die":       [0.067, 0.667, 0.067, 0.067, 0.067, 0.067],
    "Heavenly Kingdom die":     [0.368, 0.105, 0.105, 0.105, 0.105, 0.211],
    "Holy Trinity die":         [0.182, 0.227, 0.455, 0.045, 0.045, 0.045],
    "Hugo's Die":               [0.167]*6,
    "King's die":               [0.125, 0.188, 0.219, 0.25,  0.125, 0.094],
    "Lousy gambler's die":      [0.1,   0.15,  0.1,   0.15,  0.35,  0.15],
    "Lu die":                   [0.13,  0.13,  0.13,  0.13,  0.13,  0.348],
    "Lucky Die":                [0.273, 0.045, 0.091, 0.136, 0.182, 0.273],
    "Mathematician's Die":      [0.167, 0.208, 0.25,  0.292, 0.042, 0.042],
    "Molar die":                [0.167]*6,
    "Odd die":                  [0.267, 0.067, 0.267, 0.067, 0.267, 0.067],
    "Ordinary die":             [0.167]*6,  # Standard fair die
    "Painted die":              [0.188, 0.063, 0.063, 0.063, 0.438, 0.188],
    "Pie die":                  [0.462, 0.077, 0.231, 0.231, 0.0,   0.0],
    "Premolar die":             [0.167]*6,
    "Sad Greaser's Die":        [0.261, 0.261, 0.043, 0.043, 0.261, 0.13],
    "Saint Antiochus' die":     [0.0,   0.0,   1.0,   0.0,   0.0,   0.0],
    "Shrinking die":            [0.222, 0.111, 0.111, 0.111, 0.111, 0.333],
    "St. Stephen's die":        [0.167]*6,
    "Strip die":                [0.25,  0.125, 0.125, 0.125, 0.188, 0.188],
    "Three die":                [0.125, 0.063, 0.563, 0.063, 0.125, 0.063],
    "Unbalanced Die":           [0.25,  0.333, 0.083, 0.083, 0.167, 0.083],
    "Unlucky die":              [0.091, 0.273, 0.182, 0.182, 0.182, 0.091],
    "Wagoner's Die":            [0.056, 0.278, 0.333, 0.111, 0.111, 0.111],
    "Weighted die":             [0.667, 0.067, 0.067, 0.067, 0.067, 0.067],
    "Wisdom tooth die":         [0.167]*6,
}

scoring_table = {
    "1": 100,
    "5": 50,
    "1-2-3-4-5": 500,
    "2-3-4-5-6": 750,
    "1-2-3-4-5-6": 1500,
    "1-1-1": 1000,
    "2-2-2": 200,
    "3-3-3": 300,
    "4-4-4": 400,
    "5-5-5": 500,
    "6-6-6": 600,
    "1-1-1-1": 2000,
    "2-2-2-2": 400,
    "3-3-3-3": 600,
    "4-4-4-4": 800,
    "5-5-5-5": 1000,
    "6-6-6-6": 1200,
    "1-1-1-1-1": 4000,
    "2-2-2-2-2": 800,
    "3-3-3-3-3": 1200,
    "4-4-4-4-4": 1600,
    "5-5-5-5-5": 2000,
    "6-6-6-6-6": 2400,
    "1-1-1-1-1-1": 8000,
    "2-2-2-2-2-2": 1600,
    "3-3-3-3-3-3": 2400,
    "4-4-4-4-4-4": 3200,
    "5-5-5-5-5-5": 4000,
    "6-6-6-6-6-6": 4800,
}

##############################################
# 2. Helper Functions for Scoring Moves
##############################################

def group_score(indices, roll):
    """
    Given a set of indices from a roll (list of ints), return the score for that group if valid,
    else return None.
    """
    vals = [roll[i] for i in indices]
    sorted_vals = sorted(vals)
    if len(indices) == 6 and sorted_vals == [1,2,3,4,5,6]:
        return scoring_table["1-2-3-4-5-6"]
    if len(indices) == 5:
        if sorted_vals == [1,2,3,4,5]:
            return scoring_table["1-2-3-4-5"]
        if sorted_vals == [2,3,4,5,6]:
            return scoring_table["2-3-4-5-6"]
    if all(v == 1 for v in vals):
        return 100 * len(indices)
    if all(v == 5 for v in vals):
        return 50 * len(indices)
    if len(indices) == 1 and vals[0] not in (1,5):
        return None
    if len(indices) >= 3 and all(v == vals[0] for v in vals):
        key = "-".join([str(vals[0])] * len(indices))
        if key in scoring_table:
            return scoring_table[key]
    return None

@lru_cache(maxsize=None)
def best_partition_score(indices, roll):
    """
    Given a frozenset of indices (hashable) and a roll (tuple of ints),
    return the maximum score obtainable by partitioning the indices into valid scoring groups.
    Returns None if no valid partition exists.
    """
    if not indices:
        return 0
    best = None
    idx_list = list(indices)
    n = len(idx_list)
    for mask in range(1, 1 << n):
        subset = frozenset(idx_list[i] for i in range(n) if mask & (1 << i))
        score = group_score(subset, roll)
        if score is not None:
            remaining = indices - subset
            subscore = best_partition_score(remaining, roll)
            if subscore is not None:
                total = score + subscore
                if best is None or total > best:
                    best = total
            elif not remaining:
                if best is None or score > best:
                    best = score
    return best

def enumerate_move_options(roll):
    """
    For a given roll (list of ints), return a list of valid moves.
    Each move is a tuple (indices, immediate_score).
    """
    n = len(roll)
    moves = []
    roll_tuple = tuple(roll)
    for mask in range(1, 1 << n):
        indices = frozenset(i for i in range(n) if mask & (1 << i))
        score = best_partition_score(indices, roll_tuple)
        if score is not None:
            moves.append((indices, score))
    return moves

##############################################
# 3. State Space and Outcome Precomputation
##############################################

# Global cache for outcome data across states
outcome_cache = {}

def generate_state_space(candidate_lineup):
    """
    Generate all states (as sorted tuples) that can appear as subsets of candidate_lineup.
    """
    n = len(candidate_lineup)
    states = set()
    for mask in range(1 << n):
        state = tuple(sorted(candidate_lineup[i] for i in range(n) if mask & (1 << i)))
        states.add(state)
    return states

def precompute_outcome_data(state):
    """
    For a nonempty state (tuple of dice names), precompute outcome data.
    Uses a global cache to avoid redundant computation.
    Returns a list of tuples (p, moves) where p is the outcome probability and moves is
    a list of (immediate_score, new_state) for valid move options.
    """
    if state in outcome_cache:
        return outcome_cache[state]
    
    outcome_list = []
    n = len(state)
    # Iterate over all outcomes for the dice in the state
    for outcome in itertools.product(range(1,7), repeat=n):
        p = 1.0
        for i, face in enumerate(outcome):
            p *= dice_probabilities[state[i]][face-1]
        if p == 0:
            continue
        moves_raw = enumerate_move_options(list(outcome))
        move_options = []
        for indices, score in moves_raw:
            new_state = tuple(sorted(state[i] for i in range(n) if i not in indices))
            move_options.append((score, new_state))
        outcome_list.append((p, move_options))
    outcome_cache[state] = outcome_list
    return outcome_list

##############################################
# 4. Value Iteration for a Full Turn
##############################################

def compute_expected_turn_value(candidate_lineup, tol=1e-6, max_iter=1000):
    """
    Given a candidate_lineup (list of 6 dice names), compute the full-turn expected value
    by building the state space and solving the Bellman equations via value iteration.
    An empty state resets (hot dice) to the candidate lineup.
    """
    candidate_state = tuple(sorted(candidate_lineup))
    S = generate_state_space(candidate_lineup)
    outcome_data = {}
    for s in S:
        if s:
            outcome_data[s] = precompute_outcome_data(s)
    V_dict = {s: 0.0 for s in S}
    V_dict[()] = V_dict[candidate_state]
    
    for iteration in range(max_iter):
        newV = {}
        max_diff = 0.0
        for s in S:
            if not s:
                newV[s] = V_dict[candidate_state]
            else:
                total = 0.0
                for (p, moves) in outcome_data[s]:
                    if moves:
                        best_move_val = max(score + (V_dict[candidate_state] if not new_state else V_dict[new_state])
                                            for (score, new_state) in moves)
                    else:
                        best_move_val = 0.0
                    total += p * best_move_val
                newV[s] = total
            diff = abs(newV[s] - V_dict[s])
            if diff > max_diff:
                max_diff = diff
        V_dict = newV
        if max_diff < tol:
            # Convergence reached
            break
    return V_dict[candidate_state]

##############################################
# 5. Inventory Normalization and Candidate Generation
##############################################

def normalize_inventory(inventory):
    """
    Merge all dice with the standard fair distribution into "Ordinary die".
    """
    normalized = {}
    for die, count in inventory.items():
        if dice_probabilities[die] == dice_probabilities["Ordinary die"]:
            normalized["Ordinary die"] = normalized.get("Ordinary die", 0) + count
        else:
            normalized[die] = normalized.get(die, 0) + count
    return normalized

def generate_rigged_combinations(inventory, max_count=6):
    """
    Generate all multisets (combinations) of rigged dice from inventory (die name -> count)
    that use up to max_count dice. Returns a list of lists.
    """
    rigged_list = list(inventory.items())
    combinations = []
    def helper(i, current, remaining):
        if i == len(rigged_list):
            combinations.append(list(current))
            return
        die, available = rigged_list[i]
        for count in range(min(available, remaining) + 1):
            current.extend([die] * count)
            helper(i+1, current, remaining - count)
            for _ in range(count):
                current.pop()
    helper(0, [], max_count)
    return combinations

##############################################
# 6. Parallel Candidate Evaluation using Multiprocessing
##############################################

def evaluate_candidate(lineup):
    """
    Given a candidate lineup (list of 6 dice names), compute its expected turn value.
    Returns (lineup, value).
    """
    val = compute_expected_turn_value(lineup)
    return lineup, val

def best_rigged_selection(inventory):
    """
    Normalize the inventory, then generate all candidate selections (using 0 to 6 rigged dice).
    For each candidate, fill remaining slots (to 6 dice) with Ordinary die, and compute the full-turn expected value.
    Evaluation is done in parallel.
    Returns the best candidate lineup and its expected turn value.
    """
    inventory = normalize_inventory(inventory)
    all_combos = generate_rigged_combinations(inventory, 6)
    candidate_lineups = []
    for combo in all_combos:
        filler_count = 6 - len(combo)
        lineup = combo + ["Ordinary die"] * filler_count
        candidate_lineups.append(lineup)
    total_candidates = len(candidate_lineups)
    
    print("\n=== Starting Parallel Rigged Dice Turn Simulation ===\n")
    best_lineup = None
    best_value = -1.0
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_lineup = {executor.submit(evaluate_candidate, lineup): lineup for lineup in candidate_lineups}
        for idx, future in enumerate(as_completed(future_to_lineup), start=1):
            lineup, val = future.result()
            print("--------------------------------------------------")
            print(f"Candidate {idx} of {total_candidates}:")
            print(f"  Lineup: {lineup}")
            print(f"  Expected Turn Value: {val:.2f}")
            print("--------------------------------------------------\n")
            if val > best_value:
                best_value = val
                best_lineup = lineup

    print("\n=== Search Complete ===\n")
    return best_lineup, best_value

##############################################
# 7. Main Block
##############################################

if __name__ == '__main__':
    # Example inventory: available rigged dice and their quantities.
    inventory = {
        "Die of misfortune": 1,
        "Saint Antiochus' die": 5,
        "Shrinking die": 2,
        "Strip die": 1,
    }
    
    best_lineup, expected_turn_val = best_rigged_selection(inventory)
    
    print("**************** FINAL RESULT ****************\n")
    if best_lineup:
        print("Best 6-dice Lineup (Rigged dice + Ordinary dice):")
        for d in best_lineup:
            print("  -", d)
    print(f"\nExpected Turn Value: {expected_turn_val:.2f}")
    print("**********************************************\n")
