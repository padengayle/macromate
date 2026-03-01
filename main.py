import os
import json
import time
from agents import run_guardrails, run_meal_analysis, run_safety_check

# Configuration
IMAGE_DIR = "./images"
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def process_images():
    """
    Batch processes images using a waterfall pattern: 
    Guardrails (Fast) -> Analysis (Slow) -> Safety (Check).
    Logs latency and handles failures gracefully.
    """
    
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"[ERROR] No images found in {IMAGE_DIR}. Please check the directory structure.")
        return

    print(f"--- Starting Batch Pipeline for {len(image_files)} items ---")
    
    success_count = 0
    error_count = 0

    for i, img_file in enumerate(image_files):
        full_path = os.path.join(IMAGE_DIR, img_file)
        print(f"[{i+1}/{len(image_files)}] Processing {img_file}...", end="", flush=True)
        
        start_time = time.time()
        
        try:
            # Phase 1: Guardrails
            # Fail fast on non-food to save costs
            guard_result = run_guardrails(full_path)
            
            meal_dump = None
            safety_dump = None
            
            if not guard_result.is_food:
                print(" DETECTED NON-FOOD (Skipping analysis)", end="")
                # We log the rejection but do not process further
            else:
                # Phase 2: Analysis
                # Only run heavy inference if image passed guardrails
                meal_result = run_meal_analysis(full_path)
                meal_dump = meal_result.model_dump()
                
                # Phase 3: Compliance Check
                safety_result = run_safety_check(meal_result)
                safety_dump = safety_result.model_dump()

            # Observability: Capture Latency & Structured Logs
            final_output = {
                "image_id": img_file,
                "guardrails": guard_result.model_dump(),
                "analysis": meal_dump,
                "safety": safety_dump,
                "latency_ms": round((time.time() - start_time) * 1000)
            }
            
            # Persist Result
            output_filename = os.path.splitext(img_file)[0] + ".json"
            with open(os.path.join(RESULTS_DIR, output_filename), "w") as f:
                json.dump(final_output, f, indent=2)
                
            print(f" DONE ({final_output['latency_ms']}ms)")
            success_count += 1

        except Exception as e:
            # Graceful Degradation: Log error and continue to next item
            print(f" ERROR: {str(e)}")
            error_count += 1

    print("\n--- Pipeline Execution Complete ---")
    print(f"Processed: {success_count}")
    print(f"Failures: {error_count}")

if __name__ == "__main__":
    process_images()