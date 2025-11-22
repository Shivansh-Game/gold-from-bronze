import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from FNN_framework.dataset import JSONDataset
from torch.utils.data import DataLoader
from FNN_framework.activations import ParamHardSigmoid

# simple vectors as inputs, outputs logits
class SkeletalModel(nn.Module):
    # hidden_layers is a list of sizes with a length of the number of hidden layers, i.e [128, 64, 32]
    def __init__(self, n_inputs, n_outputs, hidden_layers, activation="none",dropout_rate=0.0):
        super(SkeletalModel, self).__init__()
        
        self.layers = nn.ModuleList()
        
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self.final_activation = activation
        if self.final_activation == "parametrichardsigmoid":
            self.final_activate = ParamHardSigmoid(L=0.0, U=100.0)
        
        self.layers.append(nn.Linear(n_inputs, hidden_layers[0]))
        
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))

        self.output_layer = nn.Linear(hidden_layers[-1], n_outputs)

    def forward(self, x, max_val=100):
        for layer in self.layers:
            x = self.dropout(F.elu(layer(x))) 
            # passes the output of the last layer into the next until the last hidden layer
            # and then passes it to the dropout 
            
        # output layer
        x = self.output_layer(x)
        
        # return activated logits if needed
        if self.final_activation == "none":
            return x
        elif self.final_activation == "clamp":
            return torch.clamp(x, 0, max_val)
        elif self.final_activation == "sigmoid":
            return torch.sigmoid(x) * max_val
        elif self.final_activation == "relu6":
            scale = max_val / 6.0 
            return F.relu6(x) * scale
        elif self.final_activation == "hardsigmoid":
            # F.hardsigmoid(x) outputs in the [0, 1] range
            return F.hardsigmoid(x) * max_val
        elif self.final_activation == "parametrichardsigmoid":
            return self.final_activate(x)
        else:
            raise ValueError(f"Unknown final_activation type: {self.final_activation}")


        
# loads trained model

def load_model(model_path):
    print(f"Loading model from {model_path}...")
    try:
        # load model data
        data = torch.load(model_path)
        
        # get model details
        n_ip = data['n_inputs']
        x_op = data['n_outputs']
        h_config = data['hidden_layers']
        loss = data['loss_type']
        activation = data.get('activation', 'none') 
        max_val = data.get('max_val', 100)
        dropout_rate = data.get('dropout_rate', 0.0)
        
        # remake model
        model = SkeletalModel(
            n_inputs=n_ip, 
            n_outputs=x_op, 
            hidden_layers=h_config,
            dropout_rate=dropout_rate,
            activation=activation
        )

        # load weights
        model.load_state_dict(data['model_state'])
        
        print("Model loaded successfully.")
        model.eval()
        return model, (loss, activation, max_val, dropout_rate)
        
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()
        
        
def json_trainer(h_config, model_path, file_path, batch_size, epochs, LR,
                 features_name, label_name, loss_type, activation="none",
                 max_val=100, dropout_rate=0.0,
                 lr_decay_step_size=None, lr_decay_gamma=None, optimizer="adam"):
    
    dataset = JSONDataset(file_path, features_name, label_name, loss_type=loss_type)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    n_ip = dataset.n_inputs
    x_op = dataset.n_outputs
    
    print(f"Data loaded: Found {len(dataset)} samples.")
    print(f"Auto-detected n_inputs: {n_ip}")
    print(f"Auto-detected n_outputs: {x_op}")
    print(f"Using loss function: {loss_type}")
    print(f"Using final activation: {activation} with max_val {max_val}")
    print(f"Using Dropout Rate: {dropout_rate}")
    print(f"Using Optimizer: {optimizer}")
    
    
    model = SkeletalModel(
        n_inputs=n_ip, 
        n_outputs=x_op, 
        hidden_layers=h_config,
        dropout_rate=dropout_rate,
        activation=activation
    )
    
    if loss_type == "mse":
        criterion = nn.MSELoss()
    elif loss_type == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    elif loss_type == "l1":
        criterion = nn.L1Loss()
    elif loss_type == "bce":
        # 0 or 1 classifications
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    
    if optimizer == "adam":    
        optimizer = optim.Adam(model.parameters(), lr=LR)
    elif optimizer == "adamW":
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.02)
    
    if lr_decay_step_size and lr_decay_gamma:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay_gamma)
        print(f"Using LR Scheduler: StepLR (step_size={lr_decay_step_size}, gamma={lr_decay_gamma})")
    else:
        scheduler = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in data_loader:
            
            if loss_type == "cross_entropy":
                # Cross_entropy expects (N) as the shape not (N,1)
                batch_y = batch_y.squeeze(1)
                
            # forward pass
            outputs = model(batch_X, max_val=max_val)
            # loss calc
            loss = criterion(outputs, batch_y)
            
            # backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # update total loss
            total_loss += loss.item()
        if scheduler:
            scheduler.step()
            
        # gives loss every 5 epochs
        if (epoch + 1) % 100 == 0:
            avg_loss = total_loss / len(data_loader)
            print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')
            
            
    print("--- Training Complete ---")
    
    model.eval()
    
    data_to_save = {
        'model_state': model.state_dict(),
        'n_inputs': n_ip,
        'n_outputs': x_op,
        'hidden_layers': h_config,
        'loss_type': loss_type,
        'activation': activation,
        'max_val': max_val,
        'dropout_rate': dropout_rate
    }
    
    try:
        torch.save(data_to_save, model_path)
        print(f"Model and config saved successfully to {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")