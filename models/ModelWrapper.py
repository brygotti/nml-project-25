
from models.SimpleLSTM import SimpleLSTM



# Add your model here
def get_model(model_name='SimpleLSTM', model_params=None, device=None):
    if model_name == 'SimpleLSTM':
        return SimpleLSTM(**model_params).to(device)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

