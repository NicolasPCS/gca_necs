import numpy as np
import torch

data = torch.load("/home/isipiran/exp/0406/car/f64befh_train_lion_B10/eval/samples_881999s1Hca646diet9.pt")

# Ver qué contiene
print(data.shape)
print(type(data))
print(data.keys())
print(data['ref'].shape)
print(data['optimizer_state'].shape)
pc = data['ref'][0].numpy()  # convertir a numpy si aún es tensor

print("Minimos (x, y, z):", np.min(pc, axis=0))
print("Maximos (x, y, z):", np.max(pc, axis=0))

if isinstance(data, dict):
    for k, v in data.items():
        print(f"{k}: {type(v)}, shape: {getattr(v, 'shape', 'N/A')}")
elif isinstance(data, list):
    print(f"Es una lista con {len(data)} elementos")
    for i, v in enumerate(data[:5]):
        print(f"Elemento {i}: {type(v)}, shape: {getattr(v, 'shape', 'N/A')}")
