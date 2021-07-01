import matplotlib.pyplot as plt
import string
import time
import numpy as np
from braket.circuits import Circuit, Gate, Observable
from braket.devices import LocalSimulator

"""
Create a GHZ state
Taken from AWS braket examples to play with
"""
qubits = 15
shots = 1000


def ghz_circuit(qubits):
    """
    function to return a GHZ circuit ansatz
    input: number of qubits
    """

    circuit = Circuit()

    # add Hadamard gate on first qubit
    circuit.h(0)

    # apply series of CNOT gates
    for i in range(0, qubits-1):
        circuit.cnot(control=i, target=i+1)

    return circuit


ghz = ghz_circuit(qubits)
device = LocalSimulator()
print(ghz)
result = device.run(ghz, shots=shots).result()
counts = result.measurement_counts
plt.bar(counts.keys(), counts.values())
plt.title(f'GHZ State: {qubits} Qubits - {shots} shots')
plt.xlabel('Outcomes')
plt.ylabel('Count')
plt.savefig('ghz.png')
