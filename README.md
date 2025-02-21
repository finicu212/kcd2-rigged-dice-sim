# kcd2-rigged-dice-sim
calculate best rigged dice combination based on your inventory

### NOTE: COMPUTATIONALLY INTENSIVE

Example! For inventory
```
    inventory = {
        "Die of misfortune": 1,
        "Hugo's Die": 1,
        "Molar die": 1,
        "Painted die": 1,
        "Premolar die": 1,
        "Saint Antiochus' die": 2,
        "Shrinking die": 2,
        "Strip die": 2,
        "Unbalanced Die": 1,
        "Wisdom tooth die": 1,
    }
```

Output:
"""
--------------------------------------------------
Candidate 539 of 540:
  Lineup: ['Die of misfortune', 'Ordinary die', 'Ordinary die', 'Ordinary die', 'Ordinary die', 'Painted die']
  Expected Turn Value: 771.55
--------------------------------------------------

--------------------------------------------------
Candidate 540 of 540:
  Lineup: ['Die of misfortune', 'Ordinary die', 'Ordinary die', 'Ordinary die', 'Painted die', 'Strip die']
  Expected Turn Value: 816.20
--------------------------------------------------


=== Search Complete ===

**************** FINAL RESULT ****************

Best 6-dice Lineup (Rigged dice + Ordinary dice):
  - Painted die
  - Shrinking die
  - Shrinking die
  - Strip die
  - Strip die
  - Unbalanced Die

Expected Turn Value: 1069.67
**********************************************
"""
