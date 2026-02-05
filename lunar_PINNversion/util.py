import torch
import torch.nn as nn


def select_device(verbose: bool = False) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Select GPU
        if verbose:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")  # Fallback to CPU
        if verbose:
            print("Using CPU")
    return device

# Laplacian computation
def laplacian(phi, coords):
    grads = torch.autograd.grad(
        phi,
        coords,
        grad_outputs=torch.ones_like(phi),
        create_graph=True,
    )[0]

    d2 = []

    for i in range(coords.shape[1]):
        grad2 = torch.autograd.grad(
            grads[:, i],
            coords,
            grad_outputs=torch.ones_like(grads[:, i]),
            create_graph=True,
        )[0][:, i]
        
        d2.append(grad2)
    return sum(d2)


if __name__ == "__main__":
    pass
