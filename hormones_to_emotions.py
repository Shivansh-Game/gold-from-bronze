import json
from FNN_framework.skeletalmodel import load_model
import torch


def get_response_hormones(inp, model, params):
    # run model
    with torch.no_grad():
        pred = model(inp, max_val=params)
    return pred
    


def hormones_to_emotion(oxytocin, serotonin, cortisol, adrenaline, dopamine):
    

    # normalize all inputs to a 0.0 - 1.0 scale for stable math.
    o = oxytocin / 100.0
    s = serotonin / 100.0
    c = cortisol / 100.0
    a = adrenaline / 100.0
    d = dopamine / 100.0

    # 1. HAPPY: (Well-being + Reward) *GATED BY* (NOT Stressed)
    # You can't be happy if you're stressed.
    happy_base = (s + 0.3*d) / 1.3 # normalize drivers (s+d)
    happy = happy_base * (1.0 - c)

    # 2. SAD: (Stress) *GATED BY* (NOT Happy)
    # You can't be sad if your serotonin is high.
    sad = c * (1.0 - s)
    
    # 3. ANGRY: (Threat + Stress) *GATED BY* (NOT Bonded)
    # It's hard to stay angry at someone you're bonded with.
    anger_base = (a + 0.7*c) / 1.7 # normalize drivers (a+c)
    angry = anger_base

    # 4. LOVE: (Bonding *AND* Well-being) *GATED BY* (NOT Stressed/Scared)
    # This is the "perfect" state. Requires *both* bonding AND well-being.
    # A single negative hormone will crush it.
    love_base = o * s  # multiplicative
    love_suppressor = max(c, a)
    love = love_base * (1.0 - love_suppressor)

    # 5. CARING: (Bonding) *GATED BY* (NOT in Panic)
    # This is the "pro-social" / "protective" state.
    # It's driven by the bond and *only* suppressed by high-panic (adrenaline).
    caring = o * (1.0 - a)

    emotions = {
        "happy": happy,
        "sad": sad,
        "angry": angry,
        "love": love,
        "caring": caring
    }

    # Scale 0.0-1.0 results back up to 0-100 and clip
    emotions_norm = {k: max(0, min(v * 100, 100)) for k, v in emotions.items()}

    # dominant emotion is still the max one
    dominant = max(emotions_norm, key=emotions_norm.get)
    intensity = emotions_norm[dominant]

    # If no emotion is strong enough (e.g., all are < 15),
    # just default to "neutral". This prevents "sad (2.5)" moods.
    if intensity < 30: # 30 to prevent very intense switches
        return "neutral", 0
    # -------------------------------------------

    return dominant, intensity

def logits_top3_emotions(logits_dict):
    """
    Takes the logits dictionary and returns:
      - top 3 emotions
      - each with intensity scaled to 0-100
    """

    # Convert into list of (emotion, intensity) pairs
    items = [(k, v * 100) for k, v in logits_dict.items()]

    # Sort descending by intensity
    items.sort(key=lambda x: x[1], reverse=True)

    # Return top 3
    return items[:3]

INPUT_JSON = "datasets\\emotional_transition_dataset_gemini.json"
OUTPUT_TXT = "samples\\emotion_output.txt"


def process_dataset(model, params):
    # Load dataset
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    # Open output file
    with open(OUTPUT_TXT, "w") as out:

        for idx, sample in enumerate(data):
            out.write(f"================= SAMPLE {idx+1} =================\n")


            # 1. Current emotion (from hormones)

            h = sample["inputs"]["hormones"]
            hormone_keys = ["oxytocin", "serotonin", "cortisol", "adrenaline", "dopamine"]

            curr_emotion, curr_intensity = hormones_to_emotion(
                h["oxytocin"], h["serotonin"], h["cortisol"],
                h["adrenaline"], h["dopamine"]
            )

            out.write(f"current_emotion: {curr_emotion} ({curr_intensity:.2f})\n")


            # 2. User emotion from logits (top 3)

            logits = sample["inputs"]["logits"]
            logit_keys = ["user_caring", "user_loving", "user_feeling_bad", "user_happy", 
              "user_angry", "user_thankful", "user_sorry", "neutral"]
            top3_user = logits_top3_emotions(logits)

            out.write("user_top3_emotions:\n")
            for emo, inten in top3_user:
                out.write(f"  {emo}: {inten:.2f}\n")


            # 3. Predicted next emotion

            inp = [h[key] for key in hormone_keys] + [logits[key] for key in logit_keys]
            nh = get_response_hormones(torch.tensor(inp), model, params).tolist()

            next_emotion, next_intensity = hormones_to_emotion(
                nh[0], nh[1], nh[2],
                nh[3], nh[4]
            )

            out.write(f"next_emotion: {next_emotion} ({next_intensity:.2f})\n")

            # Spacing between samples
            out.write("\n")

    print(f"Done. Saved results to: {OUTPUT_TXT}")


# Run the whole pipeline
if __name__ == "__main__":
    model, params = load_model("models\\Emotional_model.pth") # returns a tuple of model, (loss, activation, max_val)
    process_dataset(model, params[2])
