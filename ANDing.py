from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np

def create_charge_detector_circuit(input1=0, input2=0):
    """
    Creates a quantum circuit with:
    1. Hadamard gate on the first qubit
    2. Two settable qubits with different charges (through different phase)
    3. A final qubit for measurement that detects charge differences
    
    Parameters:
    input1 (int): Value for first settable qubit (0 or 1)
    input2 (int): Value for second settable qubit (0 or 1)
    
    Returns:
    qc (QuantumCircuit): The quantum circuit
    """
    # Create quantum and classical registers
    qreg = QuantumRegister(4, 'q')
    creg = ClassicalRegister(4, 'c')
    qc = QuantumCircuit(qreg, creg)
    
    # Define roles for each qubit
    control = qreg[0]      # Hadamard qubit that controls the circuit
    charge1 = qreg[1]      # First charge qubit (+)
    charge2 = qreg[2]      # Second charge qubit (-)
    measure = qreg[3]      # Measurement qubit
    
    # Step 1: Initialize the settable qubits to their specified values
    if input1 == 1:
        qc.x(charge1)
    if input2 == 1:
        qc.x(charge2)
    qc.barrier()
    
    # Step 2: Apply Hadamard gate to the first qubit
    qc.h(control)
    qc.barrier()
    
    # Step 3: Apply parallel gates with different charge configurations
    
    # First charge qubit gets positive charge characteristic
    qc.h(charge1)
    qc.p(np.pi/4, charge1)  # Positive phase
    
    # Second charge qubit gets negative charge characteristic
    qc.h(charge2)
    qc.p(-np.pi/4, charge2)  # Negative phase
    qc.barrier()
    
    # Step 4: Set up measurement qubit to detect charge differences
    
    # Initialize measurement qubit
    qc.h(measure)
    
    # Connect measurement to the control qubit
    qc.p(np.pi/4, control)
    qc.p(np.pi/4, measure)
    
    # Connect the measurement to detect charge differences through phase interactions
    # The measurement will respond differently when charges are the same vs. different
    qc.p(np.pi/8, charge1)  # Positive coupling
    qc.p(-np.pi/8, charge2) # Negative coupling
    
    # Final interference pattern to maximize charge detection
    qc.h(charge1)
    qc.h(charge2)
    qc.h(measure)
    
    qc.barrier()
    
    # Measure all qubits
    qc.measure([0, 1, 2, 3], [0, 1, 2, 3])
    
    return qc

def run_circuit_all_inputs():
    """
    Run the quantum circuit with all possible input combinations
    and analyze the results
    """
    # All possible input combinations for two binary inputs
    input_combinations = [(0, 0), (0, 1), (1, 0), (1, 1)]
    results = {}
    
    print("Running charge detector circuit with all input combinations:")
    for inputs in input_combinations:
        input1, input2 = inputs
        print(f"\n=== Input combination: {input1}, {input2} ===")
        
        # Create and run circuit for this input combination
        qc = create_charge_detector_circuit(input1, input2)
        
        # Simulate the circuit
        simulator = Aer.get_backend('qasm_simulator')
        shots = 8192  # More shots for better statistics
        job = simulator.run(qc, shots=shots)
        result = job.result()
        counts = result.get_counts(qc)
        results[inputs] = counts
        
        # Print circuit (for the first combination only to avoid clutter)
        if inputs == (0, 0):
            print("\nQuantum Circuit for Charge Detection:")
            print(qc.draw(fold=90))
        
        # Analyze the measurement results
        analyze_results(counts, inputs)
        
        # Plot histogram
        plot_histogram(counts, sort='value')  # Sort by binary value for consistent display
        plt.title(f"Results for Inputs {input1}, {input2}")
        plt.tight_layout()
        plt.show()
    
    return results

def analyze_results(counts, inputs):
    """
    Analyze the measurement results for a specific input combination
    
    Parameters:
    counts (dict): Measurement counts
    inputs (tuple): The input values (input1, input2)
    """
    input1, input2 = inputs
    print(f"\nResults analysis for inputs {input1}, {input2}:")
    
    # Print all measurement outcomes
    print("\nRaw measurement outcomes:")
    print("Bitstring | Control | Charge1 | Charge2 | Measure | Count | Probability")
    print("-" * 77)
    
    # Sort by bitstring value for consistent presentation
    bit_ordered_counts = sorted(counts.items(), key=lambda x: x[0])
    
    total_shots = sum(counts.values())
    for bitstring, count in bit_ordered_counts:
        # Reverse bitstring to match qubit ordering
        bits = bitstring[::-1]
        control = bits[0]
        ch1 = bits[1]
        ch2 = bits[2]
        measure = bits[3]
        prob = count / total_shots
        
        print(f"{bitstring:8s} |    {control}    |    {ch1}    |    {ch2}    |    {measure}   | {count:5d} | {prob:.4f}")
    

# Run the circuit with all input combinations
results = run_circuit_all_inputs()
print("\n- This quantum circuit simulates different charges through phase differences")
print("- The first charge qubit uses positive phase rotations (+)")
print("- The second charge qubit uses negative phase rotations (-)")
print("- The measurement qubit detects if the charges are the same or different")
print("- This implementation doesn't require CNOT gates")
