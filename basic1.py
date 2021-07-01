import matplotlib.pyplot as plt
from braket.circuits import Circuit
from braket.devices import LocalSimulator


# Bell State
bell = Circuit().h(0).cnot(control=0, target=1)
device = LocalSimulator()
shots = 1000
result = device.run(bell, shots=shots).result()
counts = result.measurement_counts
plt.bar(counts.keys(), counts.values())
plt.title(f'Bell State - {shots} shots')
plt.xlabel('Outcomes')
plt.ylabel('Count')
plt.savefig('bell.png')
