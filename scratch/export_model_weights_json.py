import os
import sys
import json
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.model import DiagnosticModel

def export_weights():
    models_dir = os.path.join(project_root, 'models')
    pt_path = os.path.join(models_dir, 'global_model.pt')

    model = DiagnosticModel(input_size=11, hidden_size1=64, hidden_size2=32)
    model.load_state_dict(torch.load(pt_path, map_location=torch.device('cpu')))
    model.eval()

    state = model.state_dict()
    weights_dict = {
        "fc1_w": state["fc1.weight"].tolist(),
        "fc1_b": state["fc1.bias"].tolist(),
        "fc2_w": state["fc2.weight"].tolist(),
        "fc2_b": state["fc2.bias"].tolist(),
        "fc3_w": state["fc3.weight"].tolist(),
        "fc3_b": state["fc3.bias"].tolist(),
    }

    out_path = os.path.join(models_dir, 'global_model_weights.json')
    with open(out_path, 'w') as f:
        json.dump(weights_dict, f)
    print(f"Exported model weights JSON to '{out_path}' ({os.path.getsize(out_path)} bytes)")

    # Copy to app/ and android assets
    import shutil
    shutil.copy(out_path, os.path.join('app', 'global_model_weights.json'))
    shutil.copy(out_path, os.path.join('android', 'app', 'src', 'main', 'assets', 'global_model_weights.json'))
    print("Copied model weights JSON to app/ and android assets.")

if __name__ == '__main__':
    export_weights()
