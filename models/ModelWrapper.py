from models.SimpleLSTM import SimpleLSTM
from models.TemporalBiLSTM import TemporalBiLSTM
from models.TemporalLSTMCombinedChannels import TemporalLSTMCombinedChannels
from torchinfo import summary


# Add your model here
print_summary = True
def get_model(model_name='SimpleLSTM', model_params=None, device=None):
    global print_summary
    if model_name == 'SimpleLSTM':
        model = SimpleLSTM(**model_params).to(device)
    elif model_name == 'TemporalBiLSTM':
        model = TemporalBiLSTM(**model_params).to(device)
    elif model_name == 'TemporalLSTMCombinedChannels':
        model = TemporalLSTMCombinedChannels(**model_params).to(device)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    if print_summary:
        print(f"Model summary for {model_name}:")
        print(summary(model))
        print_summary = False  # Set to False after the first call to avoid repeated printing
    return model

