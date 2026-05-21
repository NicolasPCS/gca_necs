#!/usr/bin/env python3
import os
import sys
import torch
import yaml

# Agregar la raíz del proyecto al sys.path para evitar ModuleNotFoundError
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import MinkowskiEngine as ME  # type: ignore
from models import MODEL


def create_single_voxel_seed(in_channels: int, device: torch.device) -> ME.SparseTensor:
    """Crea un SparseTensor inicial de MinkowskiEngine con un único vóxel en el origen.

    Las coordenadas de MinkowskiEngine siguen el formato: [batch_index, x, y, z]
    """
    # Inicializar vóxel semilla en el origen (0, 0, 0, 0)
    coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32).to(device)
    feats = torch.ones((1, in_channels), dtype=torch.float32).to(device)

    # Inicialización dinámica de MinkowskiEngine
    s_inicial = ME.SparseTensor(features=feats, coordinates=coords, device=device)
    return s_inicial


def main():
    # 1. Definir rutas base
    config_path = "/home/isipiran/gca_necs/log/gca_generation_airplane_30/config.yaml"
    checkpoint_path = "/home/isipiran/gca_necs/log/gca_generation_airplane_30/ckpts/ckpt-step-200000"
    output_dir = os.path.join(os.path.dirname(config_path), "generated_objs_from_single_seed")
    os.makedirs(output_dir, exist_ok=True)

    # 2. Cargar configuración y entorno de hardware
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # 3. Inicializar y cargar el modelo
    print("Cargando modelo y pesos...")
    model = MODEL[config["model"]](config, writer=None)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    # 4. Configurar parámetros de generación del autómata
    num_trials = 1
    num_steps = config.get("max_eval_steps", config.get("max_phase", 30))
    test_sample_nums = config.get("test_sample_nums", [2048])
    in_channels = config["backbone"].get("in_channels", 1)
    voxel_overflow_limit = config.get("voxel_overflow", 20000)

    print("Iniciando generación...")

    with torch.no_grad():
        for trial in range(num_trials):
            print(f"\n--- Ejecutando Trial {trial} ---")
            
            # Inicializar el autómata celular 3D con un único vóxel
            s = create_single_voxel_seed(in_channels, device)

            # Bucle de transiciones secuenciales del autómata
            for t in range(num_steps):
                s = model.transition(s)
                n_voxels = s.C.shape[0]
                print(f"    Paso {t:02d}: {n_voxels} vóxeles activos")

                # Controlar crecimiento desmedido en memoria
                if n_voxels > voxel_overflow_limit:
                    print(f"  [AVISO] Desbordamiento detectado ({n_voxels} > {voxel_overflow_limit}). Abortando transiciones.")
                    break

            print("Extrayendo nube de puntos y generando malla final...")
            s_pc_dict, mesh_dict = model.get_pointcloud(s, test_sample_nums, return_mesh=True)

            # 5. Exportar mallas resultantes
            for k, meshes in mesh_dict.items():
                for batch_idx, mesh in enumerate(meshes):
                    file_name = f"generated_trial{trial}_{k}_{batch_idx}.obj"
                    file_path = os.path.join(output_dir, file_name)
                    
                    # Trimesh export
                    mesh.export(file_path)
                    print(f"¡Hecho! Guardado exitosamente en: {file_path}")


if __name__ == "__main__":
    main()