import torch
import torch.nn as nn
from FNN_framework.skeletalmodel import json_trainer, load_model
from FNN_framework.dataset import JSONDataset

json_trainer(
    h_config=[320],
    model_path="models\\Emotional_model.pth",
    file_path="datasets\\training_data_split.json",
    batch_size=180,
    epochs=4000,
    LR=0.001,
    features_name="features",
    label_name="label",
    loss_type="l1",
    activation="parametrichardsigmoid",
    max_val=100,
    dropout_rate=0.45,
    lr_decay_step_size=2000,
    lr_decay_gamma=0.1,
    optimizer="adamW"
)
model, params = load_model("models\\Emotional_model.pth") # returns a tuple of model, (loss, activation, max_val)
loss_type = params[0]

try:
    test_data = JSONDataset("datasets\\testing_data_split.json", "features", "label", loss_type)
except Exception as e:
    print(f"Error loading or processing test file: {e}")

# run model
with torch.no_grad():
    predictions = model(test_data.X, max_val=params[2])

print("\nModel Predictions vs. Actual Labels:")
print("------------------------------------------")
input_key = ["oxytocin", "serotonin", "cortisol", "adrenaline", "dopamine", "user_caring", "user_loving", "user_feeling_bad", "user_happy", "user_angry", "user_thankful", "user_sorry", "neutral"]

output_path = "samples\\samples.txt"

with open(output_path, "w") as f:
    for idx in range(len(predictions)):
        pred = predictions[idx]
        label = test_data.y[idx]
        
        input_features = test_data.X[idx].tolist()
        
        f.write(f"Sample {idx + 1}:\n")
        
        for j, input_name in enumerate(input_key):
            f.write(f"{input_name}: {input_features[j]:.2f}\n")
        
        f.write(f"  Prediction: {pred}\n")
        f.write(f"  Actual:     {label}\n\n")

print(f"Saved results to {output_path}")


mae_criterion = nn.L1Loss()
total_mae = mae_criterion(predictions, test_data.y)

print("------------------------------------------")
print(f"Validation Complete.")
print(f"Total Mean Absolute Error (MAE): {total_mae.item():.4f}")
print("------------------------------------------")