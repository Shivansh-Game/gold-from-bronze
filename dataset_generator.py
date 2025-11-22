import json
import csv
import random
import math
import os
import time
import google.generativeai as genai
from tqdm import tqdm

# configuration

OUTPUT_FILE = "datasets\\emotional_transition_dataset_gemini.json"
TOTAL_SAMPLES_TO_GENERATE = 1000 # total samples you want in the end
BATCH_SIZE = 10                    # how many samples to generate per API call
MODEL_NAME = "gemini-2.5-flash" # Use this (10 RPM/ 250 RPD)
# MODEL_NAME = "gemini-2.5-pro" # Or this (5 RPM / 100 RPD)

# gemini-2.5-flash: 10 RPM -> 60s / 10 = 6s per request. We add a buffer.
# gemini-2.5-pro: 5 RPM -> 60s / 5 = 12s per request.
# The 100 RPD limit on Pro is the real problem, hence we stick with Flash.
SLEEP_TIME_PER_BATCH = 6.1 # (To stay under 10 RPM)

HORMONE_KEYS = ["oxytocin", "serotonin", "cortisol", "adrenaline", "dopamine"]
INTENT_KEYS = [
    "user_caring", "user_loving", "user_feeling_bad", "user_happy",
    "user_angry", "user_thankful", "user_sorry", "neutral"
]

SYSTEM_PROMPT = f"""You are an expert AI emotional state modeler. Your task is to generate a list of {BATCH_SIZE} independent, high-quality JSON objects.

Respond ONLY with a valid JSON list (an array of objects) `[ ... ]`.
Do not include any other text, reasoning, markdown, or backticks.

Hormone Key (0-100 scale):
- oxytocin: Bonding, trust, love
- serotonin: Happiness, well-being, calm
- cortisol: Stress, anxiety, fear
- adrenaline: Fight-or-flight
- dopamine: Reward, motivation

Intent Logits Key (Softmax probabilities, 0.0 to 1.0):
- user_caring: User is showing care.
- user_loving: User is expressing love.
- user_feeling_bad: User is sad/tired.
- user_happy: User is happy/excited.
- user_angry: User is angry/attacking.
- user_thankful: User is grateful.
- user_sorry: User is apologizing.
- neutral: User is neutral/greeting.

Each JSON object in the list must predict the 'next_hormones' vector based on the 'current_hormones' and 'user_intent_logits' provided in the user prompt.
All hormone values MUST be a float between 0 and 100.
"""

# --- 2. HELPER FUNCTIONS ---

def apply_softmax(logits):
    exps = [math.exp(l) for l in logits]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]

def generate_random_hormones():
    return {key: round(random.uniform(0, 100), 2) for key in HORMONE_KEYS}

def generate_random_logits():
    logits = [random.uniform(0, 1) for _ in range(len(INTENT_KEYS))]
    logits[random.randint(0, len(INTENT_KEYS) - 1)] = random.uniform(3, 7)
    softmax_probs = apply_softmax(logits)
    return {key: round(prob, 4) for key, prob in zip(INTENT_KEYS, softmax_probs)}

def parse_llm_response(response_text):
    """
    Parses and validates the JSON from the LLM's response.
    """
    data = json.loads(response_text)
    if not isinstance(data, list):
        raise ValueError("LLM did not return a list (array) of JSON objects.")
    
    clamped_samples = []
    for sample in data: # 'sample' is {"id": 0, "next_hormones": {...}}
        hormone_data = sample.get("next_hormones") 
        if not hormone_data:
            raise ValueError("JSON object in list is missing the 'next_hormones' key")

        clamped_hormones = {}
        for key in HORMONE_KEYS:
            if key not in hormone_data: # check inside hormone_data
                raise ValueError(f"Missing hormone '{key}' in 'next_hormones' object")
            value = float(hormone_data[key]) # get from hormone_data
            clamped_hormones[key] = max(0.0, min(100.0, value))
        
        # only need to return the hormones, not the ID
        clamped_samples.append(clamped_hormones)
        
    return clamped_samples

def main():
    print("Initializing Google AI client...")
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        genai.configure(api_key=api_key)
        
        generation_config = {"response_mime_type": "application/json"}
        model = genai.GenerativeModel(
            MODEL_NAME,
            system_instruction=SYSTEM_PROMPT,
            generation_config=generation_config
        )
        print(f"Client initialized for model: {MODEL_NAME}")

    except Exception as e:
        print(f"FATAL ERROR: Could not initialize Google AI client: {e}")
        return

    print(f"Starting dataset generation for {TOTAL_SAMPLES_TO_GENERATE} samples ({BATCH_SIZE} per call)...")

    all_samples = []
    num_api_calls = TOTAL_SAMPLES_TO_GENERATE // BATCH_SIZE

    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r') as f:
                all_samples = json.load(f)
            print(f"Found existing dataset! Resuming with {len(all_samples)} samples.")
        except Exception as e:
            print(f"Found {OUTPUT_FILE}, but failed to load it. Starting fresh. Error: {e}")
            all_samples = []
    
    samples_needed = TOTAL_SAMPLES_TO_GENERATE - len(all_samples)
    if samples_needed <= 0:
        print("Dataset already has the desired number of samples. Exiting.")
        return
        
    api_calls_needed = (samples_needed + BATCH_SIZE - 1) // BATCH_SIZE # rounds up 
    print(f"Need to generate {samples_needed} more samples, making {api_calls_needed} API calls...")

    for _ in tqdm(range(api_calls_needed), desc="Generating Batches"):
        response_text = None 
        try:
            # 1. make a batch of random inputs
            input_batch = []
            for i in range(BATCH_SIZE):
                input_batch.append({
                    "id": i,
                    "current_hormones": generate_random_hormones(),
                    "user_intent_logits": generate_random_logits()
                })
            
            # 2. make the user prompt
            user_prompt = f"""Generate {BATCH_SIZE} 'next_hormones' predictions for the following {BATCH_SIZE} inputs.
Your response MUST be a valid JSON list containing exactly {BATCH_SIZE} objects, one for each input.

INPUTS:
{json.dumps(input_batch, indent=2)}
"""

            # 3. call the gemini API
            response = model.generate_content(user_prompt)
            response_text = response.text
            
            # 4. validate the response (which is a list)
            next_hormones_list = parse_llm_response(response_text)
            
            if len(next_hormones_list) != BATCH_SIZE:
                print(f"\nWarning: LLM returned {len(next_hormones_list)} samples, expected {BATCH_SIZE}. Skipping batch.")
                continue

            # 5. save the data to master list
            for i in range(BATCH_SIZE):
                sample_data = {
                    "inputs": {
                        "hormones": input_batch[i]["current_hormones"],
                        "logits": input_batch[i]["user_intent_logits"]
                    },
                    "outputs": {
                        "next_hormones": next_hormones_list[i]
                    }
                }
                all_samples.append(sample_data)

        except Exception as e:
            print(f"\nWarning: Failed to generate a batch. Error: {e}")
            if response_text:
                print(f"Problematic Response: {response_text}\n")
            else:
                print("Problematic Response: N/A (Request may have failed entirely)\n")
            # we still go to 'finally' to sleep
        finally:
            # this block runs no matter what (try, except, or continue).
            # it guarantees we respect the rate limit even if the 'try' block fails.
            time.sleep(SLEEP_TIME_PER_BATCH)
            
            # save progress incrementally
            if len(all_samples) % (BATCH_SIZE * 5) == 0: # Save every 5 successful batches
                print(f"\nSaving progress... {len(all_samples)} samples saved.")
                with open(OUTPUT_FILE, 'w') as f:
                    json.dump(all_samples, f, indent=2)

            
    # write the entire list to a JSON file at the end
    print(f"\nGeneration complete. Saving final dataset with {len(all_samples)} samples to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_samples, f, indent=2)
        
    print("Done.")

if __name__ == "__main__":
    main()