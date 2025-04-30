import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
import pickle

# Constants
n_qubits = 7
n_layers = 4
num_classes = 22  # Adjust based on your dataset
dev = qml.device("default.qubit", wires=n_qubits)

# Quantum Circuit
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# QNN Classifier
class QNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.projector = nn.Linear(7, n_qubits)
        self.q_weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3))
        self.fc1 = nn.Linear(n_qubits, 16)
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.projector(x)
        q_out = []
        for xi in x:
            qc_result = quantum_circuit(xi, self.q_weights)
            if not isinstance(qc_result, torch.Tensor):
                qc_result = torch.tensor(qc_result, dtype=torch.float32)
            q_out.append(qc_result)
        q_out = torch.stack(q_out)
        x = torch.relu(self.fc1(q_out))
        return self.fc2(x)

# Load QNN model
def load_qnn_model(model_path="qnn/qnn1_model.pth", data_path="qnn/encoded_data.pkl"):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    model = QNNClassifier()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model, data['scaler'], data['label_encoder']

# Predict function for QNN
def predict_qnn(model, scaler, label_encoder, X_input):
    """
    X_input: np.array or torch.Tensor of shape (n_samples, n_features)
    Returns: list of predicted labels
    """
    if isinstance(X_input, np.ndarray):
        X_input = torch.tensor(X_input, dtype=torch.float32)

    X_input_scaled = torch.tensor(scaler.transform(X_input), dtype=torch.float32)

    model.eval()
    preds = []
    with torch.no_grad():
        out = model(X_input_scaled)
        preds = torch.argmax(out, axis=1).cpu().numpy()

    predicted_labels = label_encoder.inverse_transform(preds)
    return predicted_labels
