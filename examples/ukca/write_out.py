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

# TODO: Remove this test code
batch_size = 10
loss = torch.nn.CrossEntropyLoss()
inputs = [
    torch.randn(batch_size, input_size)
    for input_size in input_sizes
]
output = model(*inputs)
print(f"output:\n{output}")
target = torch.zeros(batch_size, dtype=torch.long)
print(f"loss:\n{loss(output, target)}")
