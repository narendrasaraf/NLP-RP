"""
game/telemetry.py
-----------------
Tracks in-game statistics (deaths, score, reaction time) over a rolling window.
When the game requests an API update, it flushes the current window data
and resets the counters for the next window.
"""

import time

class TelemetryTracker:
    def __init__(self):
        self.reset_window()

    def reset_window(self):
        """Reset raw stats for a new measurement window."""
        self.deaths = 0
        self.retries = 0
        self.score_delta = 0.0
        self.streak = 0
        self.input_actions = 0
        self.shots_fired = 0
        self.hits_landed = 0
        
        # Reaction time tracking
        self.reaction_times = []
        
        # Window timing
        self.window_start_time = time.time()

    def record_death(self):
        self.deaths += 1
        self.retries += 1
        self.streak = min(-1, self.streak - 1)  # resets win streak, increments lose streak
        self.score_delta -= 10.0

    def record_kill(self):
        self.score_delta += 15.0
        self.streak = max(1, self.streak + 1)   # resets lose streak, increments win streak
        self.hits_landed += 1

    def record_shot(self):
        self.shots_fired += 1
        self.input_actions += 1

    def record_reaction_time(self, ms: float):
        self.reaction_times.append(ms)

    def record_move(self):
        self.input_actions += 1

    def build_payload_and_reset(self) -> dict:
        """
        Calculates the finalized averages/totals for the API, 
        then resets the window internally so the game continues smoothly.
        """
        duration = time.time() - self.window_start_time
        duration = max(duration, 0.1) # prevent div by zero
        
        avg_reaction = sum(self.reaction_times) / len(self.reaction_times) if self.reaction_times else 350.0
        input_speed = self.input_actions / duration
        
        # We simulate intent gap via accuracy: low accuracy = high intent-outcome gap!
        accuracy = self.hits_landed / self.shots_fired if self.shots_fired > 0 else 0.5
        simulated_intent_gap = 1.0 - accuracy 

        payload = {
            "telemetry": {
                "deaths": self.deaths,
                "retries": self.retries,
                "score_delta": self.score_delta,
                "streak": self.streak,
                "reaction_time_ms": round(avg_reaction, 2),
                "input_speed": round(input_speed, 2)
            },
            "nlp": {
                # We won't simulate raw text dynamically right now, 
                # but we will send our simulated intent gap
                "text": "",
                "polarity": 0.0,    # Default, will modify in game via hotkeys for demo
                "intensity": 0.0,   # Default
                "intent_gap": round(simulated_intent_gap, 2)
            }
        }
        
        self.reset_window()
        return payload
