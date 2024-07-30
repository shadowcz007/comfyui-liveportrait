import torch

device_name = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f'@@device:{device_name}')

device_providers = (
    ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if torch.cuda.is_available()
    else ["MPSExecutionProvider", "CPUExecutionProvider"]
    if torch.backends.mps.is_available()
    else ["CPUExecutionProvider"]
)