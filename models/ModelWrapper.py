from models.SimpleLSTM import SimpleLSTM
from models.TemporalBiLSTM import TemporalBiLSTM
from models.TemporalLSTMCombinedChannels import TemporalLSTMCombinedChannels
from models.GCNN import GCNN
from models.Conformer import Conformer
from models.GraphSageLSTM import GraphSageLSTM
from torchinfo import summary


# Add your model here
print_summary = True
def get_model(model_name='SimpleLSTM', model_params=None, device=None):
    global print_summary
    if model_name == 'SimpleLSTM':
        model = SimpleLSTM(**model_params)
    elif model_name == 'TemporalBiLSTM':
        model = TemporalBiLSTM(**model_params)
    elif model_name == 'TemporalLSTMCombinedChannels':
        model = TemporalLSTMCombinedChannels(**model_params)
    elif model_name == 'GCNN':
        model = GCNN(**model_params)
    elif model_name == 'Conformer': 
        model = Conformer(**model_params)
    elif model_name == 'GraphSageLSTM': 
        model = GraphSageLSTM(**model_params)

    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    model = model.to(device)
    if print_summary:
        print(f"Model summary for {model_name}:")
        summary(model)
        print_summary = False  # Set to False after the first call to avoid repeated printing
    return model

