import torch
from mlstep.net import FCNN

input_sizes = (
    9,   # scalars
    83,  # jpspec
    42,  # jpdd
    34,  # jpdw
    60,  # jppj
    2,   # rchet
)
model = FCNN(*input_sizes, max_nhsteps=5, hidden_size=50)
# torch.save(model, "mlstep_model_pytorch.pt")
scripted_model = torch.jit.script(model)
scripted_model.save("mlstep_model_torchscript.pt")
