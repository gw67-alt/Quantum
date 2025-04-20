# Import necessary libraries
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import numpy as np
import matplotlib.pyplot as plt

# Create a quantum circuit with 5 qubits to represent the system components:
# qubit 0: Sodium cloud (presence detector)
# qubit 1: Object (present or absent)
# qubit 2: Object release mechanism (output linked)
# qubit 3: Door mechanism (input linked)
# qubit 4: System state (active/inactive)

qc = QuantumCircuit(5, 5)

# Initialize circuit
qc.name = "Sodium-Cloud-Detection-System"

# Initialize system state to active
qc.x(4)

# Initialize the object as present (|1⟩ state)
qc.x(1)

# Initialize the sodium cloud detector in a superposition
# (representing the probabilistic nature of the sodium cloud detection)
qc.h(0)

# Link the object release mechanism (output) with the system state
# When system is active, the release mechanism can function
qc.cx(4, 2)

# Phase 1: Object presence affects the sodium cloud detector
# If object is present (|1⟩), it interacts with the sodium cloud
qc.cz(1, 0)

# Phase 2: Simulate the object being released/removed
# Apply an X gate to qubit 1 to change its state from present to absent
qc.barrier()
print("Phase 2: Object is released/removed")
qc.x(1)  # Object becomes absent

# Phase 3: Sodium cloud detector responds to object absence
# The detector's state is affected by object's absence
qc.barrier()
print("Phase 3: Detector responds to object absence")
qc.cx(1, 0)  # Object absence affects detector

# Phase 4: Detector triggers the door mechanism (input linked)
# When detector registers object absence (|1⟩ in our encoding), door closes
qc.barrier()
print("Phase 4: Door triggered based on detector")
qc.cx(0, 3)  # Detector state controls door

# Phase 5: Check if release mechanism and door are in correct states
# Use Toffoli gate to verify: if object is absent AND door is closed,
# then the system has functioned correctly
qc.barrier()
print("Phase 5: Verifying system function")
# Measurement of the final state
qc.measure([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])

# Draw the circuit
print(qc.draw(output='text'))

# Run the simulation
simulator = AerSimulator()
compiled_circuit = transpile(qc, simulator)
job = simulator.run(compiled_circuit, shots=1024)
result = job.result()

# Get and analyze the results
counts = result.get_counts()
print("\nSimulation Results:", counts)

# Interpret the results
print("\nInterpretation of results:")
for bitstring, count in counts.items():
    # Reverse the bitstring to match qubit order (system, door, release, object, detector)
    bits = bitstring[::-1]
    detector_state = "Triggered" if bits[0] == "1" else "Not triggered"
    object_state = "Absent" if bits[1] == "0" else "Present"
    release_state = "Active" if bits[2] == "1" else "Inactive"
    door_state = "Closed" if bits[3] == "1" else "Open"
    system_state = "Active" if bits[4] == "1" else "Inactive"
    
    probability = count / 1024 * 100
    print(f"Scenario (prob: {probability:.1f}%): Sodium detector: {detector_state}, Object: {object_state}, "
          f"Release mechanism: {release_state}, Door: {door_state}, System: {system_state}")

# Plot the histogram of results
fig = plot_histogram(counts)
# fig.savefig('sodium_detection_results.png')
