import json
import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_FILE = "datasets\\emotional_transition_dataset_gemini.json"
OUTPUT_FILE = "datasets\\flattened_dataset_for_fnn.json"

# these lists guarantee the order of your features is ALWAYS the same.
HORMONE_KEYS = ["oxytocin", "serotonin", "cortisol", "adrenaline", "dopamine"]
INTENT_KEYS = [
    "user_caring", "user_loving", "user_feeling_bad", "user_happy",
    "user_angry", "user_thankful", "user_sorry", "neutral"
]
oxy = HORMONE_KEYS[0]
cort = HORMONE_KEYS[2]
sero = HORMONE_KEYS[1]
dopa = HORMONE_KEYS[4]

loving_logit = INTENT_KEYS[1]
angry_logit = INTENT_KEYS[4]
bad_logit = INTENT_KEYS[2]
happy_logit = INTENT_KEYS[3]

print(f"Loading complex dataset from {INPUT_FILE}...")
with open(INPUT_FILE, 'r') as f:
    complex_data = json.load(f)

flattened_data = []
for sample in complex_data:
    # 1. flatten the 13 inputs (5 hormones + 8 logits)
    features_list = []
    for key in HORMONE_KEYS:
        features_list.append(sample["inputs"]["hormones"][key])
    for key in INTENT_KEYS:
        features_list.append(sample["inputs"]["logits"][key])
        
    # adding new features to give the AI more inputs to work with, ended up being scrapped as these were just noise
    #features_list.append(sample["inputs"]["hormones"][oxy] * sample["inputs"]["logits"][loving_logit]) 
    #features_list.append(sample["inputs"]["hormones"][cort] * sample["inputs"]["logits"][angry_logit])
    #features_list.append(sample["inputs"]["hormones"][sero] * sample["inputs"]["logits"][bad_logit])
    #features_list.append(sample["inputs"]["hormones"][dopa] * sample["inputs"]["logits"][happy_logit])
    #features_list.append(sample["inputs"]["hormones"][oxy] / (sample["inputs"]["hormones"][cort] + 1)) # 1 to prevent dividing by 0

    # 2. flatten the 5 outputs (next hormones)
    label_list = []
    for key in HORMONE_KEYS:
        label_list.append(sample["outputs"]["next_hormones"][key])
    
    # 3. create the new, simple format
    flattened_data.append({
        "features": features_list,
        "label": label_list
    })

print(f"Saving {len(flattened_data)} flattened samples to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w') as f:
    json.dump(flattened_data, f, indent=2)

print("Done.")



# load the original JSON data
file_name = 'datasets\\flattened_dataset_for_fnn.json'
with open(file_name, 'r') as f:
    data = json.load(f)

# split the data. We use random_state for reproducibility.
train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

# define file names for the output
train_file_name = 'datasets\\training_data_split.json'
test_file_name = 'datasets\\testing_data_split.json'

# save the new training data split
with open(train_file_name, 'w') as f:
    json.dump(train_data, f, indent=2)

# save the new testing data split
with open(test_file_name, 'w') as f:
    json.dump(test_data, f, indent=2)

print(f"Original data had {len(data)} records.")
print(f"Successfully created '{train_file_name}' with {len(train_data)} records.")
print(f"Successfully created '{test_file_name}' with {len(test_data)} records.")