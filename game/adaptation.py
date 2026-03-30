"""
game/adaptation.py
------------------
Local game client logic that smoothly translates the generic NLP/ML backend states 
(Frustrated/Bored) into concrete Pygame physics multipliers.
"""

def calculate_new_speed(current_speed: float, state: str) -> float:
    """
    Adjusts the game speed based on player cognitive state.
    Limits speed between 0.5 (very easy) and 3.0 (very hard).
    Ensures smooth transitions (small steps rather than sudden jumps).
    """
    MIN_SPEED = 0.5
    MAX_SPEED = 3.0
    
    # Smooth transition increments
    # NOTE: Decrease is historically faster than increase in games to instantly 
    # relieve acute frustration, while boredom is cured gradually.
    INCREASE_DELTA = 0.10
    DECREASE_DELTA = 0.25 
    
    if state == "Frustrated":
        new_speed = current_speed - DECREASE_DELTA
        
    elif state == "Bored":
        new_speed = current_speed + INCREASE_DELTA
        
    else: # "Engaged" or unknown
        # Maintain current sweet-spot
        new_speed = current_speed
        
    # Clamp to strict physics limits required by Pygame
    new_speed = max(MIN_SPEED, min(MAX_SPEED, new_speed))
    
    return round(new_speed, 2)
