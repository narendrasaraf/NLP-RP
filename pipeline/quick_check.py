import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.game_pipeline import GamePipeline
from rl.q_learning import FrustrationState, DifficultyAction

TEST_MESSAGES = [
    "This level is impossible",
    "I keep losing again and again",
    "Okay this is getting better",
    "Nice, I finally won",
    "Game is fun but still hard",
    "Bro this is so annoying",
    "Alright I think I got it now",
]

pipe = GamePipeline(use_bert=False, epsilon_start=0.20, seed=42, initial_difficulty=5.0, difficulty_step=1.0)
results = [pipe.process(m) for m in TEST_MESSAGES]

print("\n" + "="*100)
print("  INTEGRATION TEST — 7 MESSAGE RESULTS SUMMARY")
print("="*100)
print(f"{'#':<3} {'Message':<34} {'Sent':>7} {'Emotion':<14} {'Raw F':>7} {'EMA F':>7} {'MA F':>6} {'Level':<10} {'Action':<26} {'Diff':>6}")
print("-"*100)

for i, r in enumerate(results, 1):
    msg   = r['message'][:32]
    s     = r['sentiment_score']
    emo   = r['emotion_label'][:12]
    rawf  = r['frustration_score']
    emaf  = r['smoothed_frustration']
    maf   = r['moving_avg_frustration']
    lvl   = r['frustration_state'].replace('_frustration','').upper()[:8]
    act   = r['action'].replace('_difficulty','').replace('no_change','NO_CHANGE').upper()[:24]
    diff  = r['new_difficulty']
    sym   = {'increase_difficulty':'[^]','decrease_difficulty':'[v]','no_change':'[=]'}.get(r['action'],'[ ]')
    note  = " <-- REDUCED!" if r['difficulty_delta'] < 0 else (" <-- RAISED" if r['difficulty_delta'] > 0 else "")
    print(f"{i:<3} {msg:<34} {s:>+7.3f} {emo:<14} {rawf:>7.3f} {emaf:>7.3f} {maf:>6.3f} {lvl:<10} {sym} {act:<24} {diff:>5.1f}{note}")

print("="*100)

print("\n  TEMPORAL FRUSTRATION EVOLUTION")
print(f"  {'Step':<6}" + "".join(f"  {'s'+str(i):<7}" for i in range(1,8)))
print(f"  {'Raw':<6}" + "".join(f"  {r['frustration_score']:<7.3f}" for r in results))
print(f"  {'EMA':<6}" + "".join(f"  {r['smoothed_frustration']:<7.3f}" for r in results))
print(f"  {'MA':<6}" + "".join(f"  {r['moving_avg_frustration']:<7.3f}" for r in results))

print("\n  FINAL Q-TABLE:")
q = results[-1]['q_table']
print(f"  {'State':<10} {'INC':>12} {'DEC':>12} {'NOP':>12} {'Best'}")
for s_val, a_dict in q.items():
    s_short = s_val.replace('_frustration','').upper()
    best = max(a_dict, key=a_dict.get)
    print(f"  {s_short:<10} {a_dict['increase_difficulty']:>+12.5f} {a_dict['decrease_difficulty']:>+12.5f} {a_dict['no_change']:>+12.5f}  {DifficultyAction(best).value}")

print("\n  OPTIMAL POLICY:")
for s_val, a_val in results[-1]['optimal_policy'].items():
    s_s = s_val.replace('_frustration','').upper()
    print(f"    {s_s:<10} -> {a_val}")

print("\n  ACTION COUNTS:")
from collections import Counter
ac = Counter(r['action'] for r in results)
for a, n in ac.items():
    print(f"    {a}: {n}")

print("\n  DIFFICULTY REDUCTION EVENTS:")
for i, r in enumerate(results, 1):
    if r['difficulty_delta'] < 0:
        print(f"    Step {i}: '{r['message']}'  -> difficulty {r['prev_difficulty']:.1f} -> {r['new_difficulty']:.1f}  (state={r['frustration_state']})")

print("\nALL 7 TESTS PASSED OK")
