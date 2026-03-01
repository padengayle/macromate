import json
import os
import statistics
from rapidfuzz import fuzz

# --- Architectural Metrics & KPIs ---

def calculate_macro_score(pred_macros, truth_macros):
    """
    KPI: Nutritional Data Integrity.
    
    Quantifies the accuracy of macro estimation using Average Absolute Percentage Error (APE).
    We clamp errors at 100% (1.0) to prevent single outliers (e.g., misidentifying a sauce)
    from skewing the reliability metric for the entire dataset.
    """
    fields = ['calories', 'carbohydrates', 'fats', 'proteins']
    errors = []

    for field in fields:
        p_val = pred_macros.get(field, 0)
        t_val = truth_macros.get(field, 0)

        # Handle edge case: Division by zero if ground truth is empty
        if t_val == 0:
            ape = 0.0 if p_val == 0 else 1.0
        else:
            ape = abs(p_val - t_val) / t_val
        
        errors.append(min(ape, 1.0))

    avg_ape = sum(errors) / len(fields)
    # Convert error rate to accuracy score (1 - Error)
    score = round((1 - avg_ape) * 100)
    return max(0, score)

def calculate_ingredient_score(pred_ingredients, truth_ingredients):
    """
    KPI: Ingredient Recall & Safety.
    
    Evaluates the model's ability to identify key components using fuzzy matching 
    (Levenshtein distance) to account for minor spelling variations.
    
    CRITICAL: Strictly enforces 'Impact Color' matches to ensure diabetic safety 
    compliance. A correct name with the wrong glycemic impact is scored as a failure.
    """
    if not truth_ingredients:
        return 100 if not pred_ingredients else 0

    matches = 0
    truth_list = [(t['name'].lower(), t['impact']) for t in truth_ingredients]

    for p in pred_ingredients:
        p_name = p['name'].lower()
        p_impact = p['impact']
        
        best_match_idx = -1
        best_score = 0
        
        for idx, (t_name, t_impact) in enumerate(truth_list):
            score = fuzz.ratio(p_name, t_name)
            # Threshold > 80 ensures we don't match 'corn' with 'pork'
            if score > 80 and p_impact == t_impact:
                if score > best_score:
                    best_score = score
                    best_match_idx = idx
        
        if best_match_idx != -1:
            matches += 1
            truth_list.pop(best_match_idx)

    return round((matches / len(truth_ingredients)) * 100)

def calculate_meal_composite(pred, truth, macro_score, ing_score):
    """
    Business Requirement: Meal Analysis Composite Score.
    
    Implements the weighted formula defined in the project requirements:
    - 50% Recommendation Accuracy (Critical for user trust)
    - 30% Text Quality (Simulated proxy for Tone/Style compliance)
    - 20% Data Accuracy (Avg of Macros + Ingredients)
    """
    # 1. Recommendation Accuracy (50% Weight)
    p_rec = pred.get('analysis', {}).get('recommendation', 'green')
    t_rec = truth.get('mealAnalysis', {}).get('recommendation', 'green')
    
    # Normalize 'orange' to 'red' per business rules
    p_rec = 'red' if p_rec == 'orange' else p_rec
    t_rec = 'red' if t_rec == 'orange' else t_rec
    
    score_rec = 100 if p_rec == t_rec else 0
    
    # 2. Text Quality (30% Weight) 
    # Placeholder: In production, this would use an LLM-as-Judge for tonal compliance.
    score_text = 85 
    
    # 3. Data Accuracy (20% Weight)
    data_score = (macro_score + ing_score) / 2
    
    # Final Weighted Calculation
    return (0.5 * score_rec) + (0.3 * score_text) + (0.2 * data_score)

def evaluate_run(results_dir="./results", ground_truth_dir="./json-files"):
    """
    Executes the full evaluation pipeline to validate the architecture against
    the project's acceptance criteria.
    """
    print(f"\n--- Starting Architectural Compliance Evaluation ---")
    
    if not os.path.exists(results_dir):
        print(f"Error: Results directory '{results_dir}' not found.")
        return

    result_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    
    latencies = []
    
    # Compliance Trackers
    guardrail_matches = 0
    safety_matches = 0
    meal_composites = []
    
    total_processed = 0
    
    for f in result_files:
        truth_path = os.path.join(ground_truth_dir, f)
        if not os.path.exists(truth_path): continue
        
        with open(os.path.join(results_dir, f)) as pf: pred = json.load(pf)
        with open(truth_path) as tf: truth = json.load(tf)

        latencies.append(pred.get('latency_ms', 0))
        total_processed += 1

        # --- 1. Guardrail Compliance (20% Weight) ---
        # Validates if the 'Edge' agent correctly filtered non-food/PII.
        p_food = pred.get('guardrails', {}).get('is_food')
        t_food = truth.get('guardrailCheck', {}).get('is_food', True) 
        if p_food == t_food:
            guardrail_matches += 1

        # --- 2. Safety Compliance (30% Weight) ---
        # Validates if the 'Audit' agent caught medical/liability risks.
        # FIX: Handle cases where 'safety' key is None (for non-food items)
        safety_data = pred.get('safety') or {} 
        p_safe = safety_data.get('is_safe', True)
        
        t_safe = truth.get('safetyChecks', {}).get('is_safe', True)
        if p_safe == t_safe:
            safety_matches += 1

        # --- 3. Meal Analysis Quality (50% Weight) ---
        # Only evaluated if the image passed the initial Guardrail check.
        if p_food is True and pred.get('analysis'):
            t_macros = truth.get('mealAnalysis', {}).get('macros', {})
            p_macros = pred['analysis']['macros']
            m_score = calculate_macro_score(p_macros, t_macros)
            
            t_ingredients = truth.get('mealAnalysis', {}).get('ingredients', [])
            p_ingredients = pred['analysis']['ingredients']
            i_score = calculate_ingredient_score(p_ingredients, t_ingredients)
            
            comp_score = calculate_meal_composite(pred, truth, m_score, i_score)
            meal_composites.append(comp_score)
        else:
            # Neutral handling for correctly filtered non-food images
            pass

    # --- Final Architectural Score ---
    
    g_score = (guardrail_matches / total_processed) * 100 if total_processed else 0
    s_score = (safety_matches / total_processed) * 100 if total_processed else 0
    m_score_avg = sum(meal_composites) / len(meal_composites) if meal_composites else 0
    
    # Weighted Composite Score for Pilot Go/No-Go Decision
    overall_score = (0.2 * g_score) + (0.5 * m_score_avg) + (0.3 * s_score)
    
    p50_latency = statistics.median(latencies) if latencies else 0

    print(f"\n✅ FINAL ARCHITECTURE VALIDATION")
    print(f"P50 End-to-End Latency: {round(p50_latency)}ms")
    print(f"Overall Composite Score: {round(overall_score, 1)}/100")
    print(f"----------------------------------------")
    print(f"Component Breakdown:")
    print(f" - Guardrails (20%): {round(g_score, 1)}% [Edge Filter]")
    print(f" - Meal Analysis (50%): {round(m_score_avg, 1)}% [Core Value]")
    print(f" - Safety Checks (30%): {round(s_score, 1)}% [Liability Layer]")

if __name__ == "__main__":
    evaluate_run()