import time
import sys

def main():
    print("Loading predictor...")
    from backend.predictor import predictor_engine
    
    # Warm up (first call might have lazy initializations)
    predictor_engine.predict(0.0)
    
    print("Running 100 ticks...")
    start_time = time.perf_counter()
    
    for _ in range(100):
        res = predictor_engine.predict(-0.5)
        
    end_time = time.perf_counter()
    
    elapsed_ms = (end_time - start_time) * 1000
    avg_ms = elapsed_ms / 100
    
    print(f"Total time 100 ticks: {elapsed_ms:.2f} ms")
    print(f"Average time per tick: {avg_ms:.2f} ms")
    
    if avg_ms < 50:
        print("PERFORMANCE: SUCCESS (<50ms)")
    else:
        print("PERFORMANCE: FAILED (>50ms)")
        
if __name__ == "__main__":
    main()
