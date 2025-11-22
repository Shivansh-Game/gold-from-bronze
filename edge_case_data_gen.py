import json
import random

def generate_logits(dominant_emotion_index):
    # makes a dict of 8 logits, normalized to sum to 1.0
    logits = [random.uniform(0.01, 0.05) for _ in range(8)]
    logits[dominant_emotion_index] = random.uniform(0.85, 0.98)
    
    total = sum(logits)
    logits_normalized = [l / total for l in logits]
    
    logit_keys = ["user_caring", "user_loving", "user_feeling_bad", "user_happy", "user_angry", "user_thankful", "user_sorry", "neutral"]
    return dict(zip(logit_keys, logits_normalized))

def generate_hormones(preset=None):
    # makes a dict of 5 random hormone values, with some presets available
    if preset == "low_stress_low_energy":
        values = [
            random.uniform(10, 30),  # low-ish oxytocin
            random.uniform(5, 15),   # very low serotonin
            random.uniform(5, 20),   # very low cortisol
            random.uniform(5, 20),   # very low adrenaline
            random.uniform(10, 30)   # low-ish dopamine
        ]
    elif preset == "high_stress":
        values = [
            random.uniform(10, 30),  # low oxytocin
            random.uniform(10, 30),  # low serotonin
            random.uniform(80, 95),  # high cortisol
            random.uniform(80, 95),  # high adrenaline
            random.uniform(10, 30)   # low dopamine
        ]
    elif preset == "high_positive":
        values = [
            random.uniform(80, 95),  # high oxytocin
            random.uniform(80, 95),  # high serotonin
            random.uniform(5, 20),   # low cortisol
            random.uniform(5, 20),   # low adrenaline
            random.uniform(80, 95)   # high dopamine
        ]
    else: # Default: generate completely random
        values = [random.uniform(0.0, 100.0) for _ in range(5)]
        
    keys = ["oxytocin", "serotonin", "cortisol", "adrenaline", "dopamine"]
    return dict(zip(keys, values))

def apply_logic(hormones, logits):
    # applies the logical rules to generate the next hormone state
    new_hormones = hormones.copy()
    
    # find the dominant emotion
    logit_list = list(logits.values())
    emotion_index = max(range(8), key=lambda i: logit_list[i])
    emotion_strength = logit_list[emotion_index]
    
    base_change = 25.0 * emotion_strength # keeps max change at below 25 

    oxy = new_hormones["oxytocin"]
    ser = new_hormones["serotonin"]
    cor = new_hormones["cortisol"]
    adr = new_hormones["adrenaline"]
    dop = new_hormones["dopamine"]

    # Rule 1, for positive intents (caring[0], loving[1], happy[3], thankful[5])
    if emotion_index in [0, 1, 3, 5]:
        # State dependency: Harder to raise a high value, easier to raise a low value
        new_hormones["oxytocin"] += base_change * (1.0 - oxy / 100.0)
        new_hormones["serotonin"] += base_change * (1.0 - ser / 100.0)
        new_hormones["dopamine"] += base_change * (1.0 - dop / 100.0)
        # State dependency: Easy to lower a high value
        new_hormones["cortisol"] -= base_change * (cor / 100.0)
        new_hormones["adrenaline"] -= base_change * (adr / 100.0)

    # Rule 2, for negative intents (feeling_bad[2], angry[4])
    elif emotion_index in [2, 4]:
        new_hormones["oxytocin"] -= base_change * (oxy / 100.0)
        new_hormones["serotonin"] -= base_change * (ser / 100.0)
        new_hormones["dopamine"] -= base_change * (dop / 100.0)
        # State dependency: Harder to raise a high value
        new_hormones["cortisol"] += base_change * (1.0 - cor / 100.0)
        new_hormones["adrenaline"] += base_change * (1.0 - adr / 100.0)
    
    # Rule 3, for Sorry[6] (socially positive, stress-reducing)
    elif emotion_index == 6:
        new_hormones["oxytocin"] += base_change * (1.0 - oxy / 100.0)
        new_hormones["serotonin"] += base_change * (1.0 - ser / 100.0)
        new_hormones["cortisol"] -= base_change * (cor / 100.0)
        new_hormones["adrenaline"] -= base_change * (adr / 100.0)

    # Rule 4, for Neutral[7] (small random drift)
    else:
        for k in new_hormones:
            new_hormones[k] += random.uniform(-2.5, 2.5)

    # Clip all values to the valid [0, 100] range
    for k in new_hormones:
        new_hormones[k] = round(max(0.0, min(100.0, new_hormones[k])), 2)
    
    return new_hormones

# generation starts here
new_data = []

# 1. (Low-Energy + Thankful)
for _ in range(5):
    inputs = {
        "hormones": generate_hormones(preset="low_stress_low_energy"),
        "logits": generate_logits(5) # 5 = user_thankful
    }
    outputs = {"next_hormones": apply_logic(inputs["hormones"], inputs["logits"])}
    new_data.append({"inputs": inputs, "outputs": outputs})

# 2. (Low-Stress + Angry)
for _ in range(5):
    inputs = {
        "hormones": generate_hormones(preset="low_stress_low_energy"),
        "logits": generate_logits(4) # 4 = user_angry
    }
    outputs = {"next_hormones": apply_logic(inputs["hormones"], inputs["logits"])}
    new_data.append({"inputs": inputs, "outputs": outputs})

# 3. (High-Stress + Happy)
for _ in range(5):
    inputs = {
        "hormones": generate_hormones(preset="high_stress"),
        "logits": generate_logits(3) # 3 = user_happy
    }
    outputs = {"next_hormones": apply_logic(inputs["hormones"], inputs["logits"])}
    new_data.append({"inputs": inputs, "outputs": outputs})

# 4. (High-Positive + Loving)
# this is for teaching the model how to handle clipping at 100.0.
for _ in range(5):
    inputs = {
        "hormones": generate_hormones(preset="high_positive"),
        "logits": generate_logits(1) # 1 = user_loving
    }
    outputs = {"next_hormones": apply_logic(inputs["hormones"], inputs["logits"])}
    new_data.append({"inputs": inputs, "outputs": outputs})

# Convert the list of dictionaries to a JSON string
json_output = json.dumps(new_data, indent=2)
print(json_output)