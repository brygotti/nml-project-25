from models.SimpleLSTM import SimpleLSTM
from models.TemporalBiLSTM import TemporalBiLSTM
from models.TemporalLSTMCombinedChannels import TemporalLSTMCombinedChannels


# Add your model here
def get_model(model_name='SimpleLSTM', model_params=None, device=None):
    if model_name == 'SimpleLSTM':
        return SimpleLSTM(**model_params).to(device)
    elif model_name == 'TemporalBiLSTM':
        return TemporalBiLSTM(**model_params).to(device)
    elif model_name == 'TemporalLSTMCombinedChannels':
        return TemporalLSTMCombinedChannels(**model_params).to(device)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

