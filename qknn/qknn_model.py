import numpy as np
import pickle
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Amplitude encoding for quantum state preparation
def amplitude_encode(features):
    d = len(features)
    n_qubits = int(np.ceil(np.log2(d)))
    dim = 2 ** n_qubits
    vec = np.zeros(dim)
    vec[:d] = features
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec

# Prepare quantum state
def build_state_prep(vec, n_qubits):
    qc = QuantumCircuit(n_qubits)
    qc.initialize(vec, range(n_qubits))
    return qc

# Quantum similarity using the swap test
def swap_test(vec1, vec2, shots=1024):
    n_qubits = int(np.log2(len(vec1)))
    qc = QuantumCircuit(1 + 2 * n_qubits, 1)
    qc.h(0)
    qc.compose(build_state_prep(vec1, n_qubits), qubits=range(1, 1 + n_qubits), inplace=True)
    qc.compose(build_state_prep(vec2, n_qubits), qubits=range(1 + n_qubits, 1 + 2 * n_qubits), inplace=True)
    
    for i in range(n_qubits):
        qc.cswap(0, i + 1, i + 1 + n_qubits)

    qc.h(0)
    qc.measure(0, 0)

    simulator = AerSimulator()
    tqc = transpile(qc, simulator)
    job = simulator.run(tqc, shots=shots)
    result = job.result()
    counts = result.get_counts()
    prob_0 = counts.get('0', 0) / shots
    fidelity = 2 * prob_0 - 1
    return fidelity

# Load QkNN model
def load_qknn_model(model_path="qknn_model.pkl"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model["train_vecs"], model["train_labels"]

# QkNN Prediction
def predict_qknn(X_input, train_vecs, train_labels, k=3):
    X_encoded = [amplitude_encode(x) for x in X_input]
    predictions = []

    for tvec in X_encoded:
        sims = [swap_test(tvec, trvec) for trvec in train_vecs]
        knn_indices = np.argsort(sims)[-k:]
        knn_labels = [train_labels[i] for i in knn_indices]
        pred = max(set(knn_labels), key=knn_labels.count)
        predictions.append(pred)

    return predictions
