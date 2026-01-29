# GPU Monte-Carlo Rigged Dice Lineup Search for Kingdom Come Deliverance 2

A GPU-accelerated script that searches all possible 6-dice lineups from your inventory to find the best one for a Farkle-like game.

It estimates:
- Expected score per turn
- Farkle rate
- Hot-dice runaway probability (treated as infinite value if reliable)

Lineups that can repeatedly trigger hot-dice resets are detected as "infinite" and prioritized.

Buy me a coffee! https://buymeacoffee.com/finicu

## Usage

### Examples
```bash
python gpu_search.py            # default balanced mode
python gpu_search.py --quick    # fast & noisy (good for exploration)
python gpu_search.py --accurate # slow & (more) precise , note: remove bad dice from your inventory first.
python gpu_search.py -v         # verbose batch logging
```

### Help:
```
$ python .\kcd2-gpu.py -h
usage: kcd2-gpu.py [-h] [-v] [--quick] [--accurate]

GPU Monte-Carlo rigged-dice lineup search

options:
  -h, --help     show this help message and exit
  -v, --verbose  Enable verbose per-batch debug output
  --quick        Fast, noisy preset (few sims, good for exploration)
  --accurate     Slow, stable preset (many sims, reliable results)
```

## Example

```bash
# Quick scan with verbose output
python gpu_search.py --quick -v
```

This will evaluate every possible 6-dice lineup from your inventory and print the best one at the end.

## How to Configure Your Inventory

Open the script and edit the `INVENTORY` dictionary near the top:

```python
# example of a real inventory to be used with --quick (remove crap die before using --accurate)
INVENTORY: Dict[str, int] = {
    "Die of misfortune": 1,
    "Saint Antiochus' die": 1,
    "Odd die": 3,
    "Holy Trinity die": 3,
    "Shrinking die": 2,
    "Strip die": 1,
    "Wagoner's die": 4,
    "Weighted die": 1
}
```

All known rigged dice probabilities are already in the script (`ALL_DICE` dictionary). Ordinary dice are automatically used as fillers.

## Requirements

- Python 3.8+
- PyTorch (with CUDA for GPU acceleration strongly recommended)

Install with:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
(Adjust CUDA version as needed; CPU-only works but is much slower.)

## Notes for Beginners

- The script runs Monte-Carlo simulations (many random rolls) to estimate performance.
- `--quick` is great for testing many lineups fast.
- `--accurate` uses many more simulations for reliable numbers (can take minutes/hours).
- If a lineup is detected as "infinite" (high chance of endless hot-dice), the search can stop early.
- Output includes progress logs, final best lineup, and a short list of results.


## Example Search

```
python .\kcd2-gpu.py
[PRESET] Default (balanced) settings
Starting search...
[SEARCH] Candidates generated: 7. Device=cuda. Testing order prefers more rigged dice first.
[SEARCH] Evaluating chunk 1 (7 lineups)...
 [NUMERIC] ["Saint Antiochus' die", "Saint Antiochus' die", "Saint Antiochus' die", "Saint Antiochus' die", "Saint Antiochus' die", "Saint Antiochus' die"] EV=7200.00 farkle=0.000 runaway=1.000 hybrid=7200.00
 [NUMERIC] ["Saint Antiochus' die", "Saint Antiochus' die", "Saint Antiochus' die", "Saint Antiochus' die", "Saint Antiochus' die", 'Ordinary die'] EV=4287.50 farkle=0.060 runaway=0.940 hybrid=4030.25
 [NUMERIC] ["Saint Antiochus' die", "Saint Antiochus' die", "Saint Antiochus' die", "Saint Antiochus' die", 'Ordinary die', 'Ordinary die'] EV=2821.50 farkle=0.040 runaway=0.960 hybrid=2708.64
 [NUMERIC] ["Saint Antiochus' die", "Saint Antiochus' die", "Saint Antiochus' die", 'Ordinary die', 'Ordinary die', 'Ordinary die'] EV=2189.00 farkle=0.000 runaway=1.000 hybrid=2189.00
 [NUMERIC] ["Saint Antiochus' die", "Saint Antiochus' die", 'Ordinary die', 'Ordinary die', 'Ordinary die', 'Ordinary die'] EV=3523.50 farkle=0.420 runaway=0.160 hybrid=2043.63
 [NUMERIC] ["Saint Antiochus' die", 'Ordinary die', 'Ordinary die', 'Ordinary die', 'Ordinary die', 'Ordinary die'] EV=2687.50 farkle=0.000 runaway=1.000 hybrid=2687.50
 [NUMERIC] ['Ordinary die', 'Ordinary die', 'Ordinary die', 'Ordinary die', 'Ordinary die', 'Ordinary die'] EV=2880.00 farkle=0.000 runaway=1.000 hybrid=2880.00

=== FINAL RESULT ===
Best numeric lineup with hybrid score 7200.00:
["Saint Antiochus' die", "Saint Antiochus' die", "Saint Antiochus' die", "Saint Antiochus' die", "Saint Antiochus' die", "Saint Antiochus' die"]

Here are some other results:
 ["Saint Antiochus' die", "Saint Antiochus' die", "Saint Antiochus' die", "Saint Antiochus' die", "Saint Antiochus' die", "Saint Antiochus' die"] -> 7200.00
 ["Saint Antiochus' die", "Saint Antiochus' die", "Saint Antiochus' die", "Saint Antiochus' die", "Saint Antiochus' die", 'Ordinary die'] -> 4030.25
 ["Saint Antiochus' die", "Saint Antiochus' die", "Saint Antiochus' die", "Saint Antiochus' die", 'Ordinary die', 'Ordinary die'] -> 2708.64
 ["Saint Antiochus' die", "Saint Antiochus' die", "Saint Antiochus' die", 'Ordinary die', 'Ordinary die', 'Ordinary die'] -> 2189.00
 ["Saint Antiochus' die", "Saint Antiochus' die", 'Ordinary die', 'Ordinary die', 'Ordinary die', 'Ordinary die'] -> 2043.63
 ["Saint Antiochus' die", 'Ordinary die', 'Ordinary die', 'Ordinary die', 'Ordinary die', 'Ordinary die'] -> 2687.50
 ['Ordinary die', 'Ordinary die', 'Ordinary die', 'Ordinary die', 'Ordinary die', 'Ordinary die'] -> 2880.00
```
