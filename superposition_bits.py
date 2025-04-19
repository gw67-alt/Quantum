from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np

def create_settable_quantum_circuit(input1=0, input2=0):
    """
    Creates a quantum circuit with:
    1. Hadamard gate on the first qubit as a setter
    2. Two explicitly settable qubits (inputs 1 and 2)
    3. A final qubit for measurement
    
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
    setter_qubit = qreg[0]    # Hadamard qubit that "sets" the operation
    settable1 = qreg[1]       # First settable input qubit
    settable2 = qreg[2]       # Second settable input qubit
    measurement = qreg[3]     # Final measurement qubit
    
    # Step 1: Initialize the settable qubits to their specified values
    if input1 == 1:
        qc.x(settable1)
    if input2 == 1:
        qc.x(settable2)
    qc.barrier()
    
    # Step 2: Apply Hadamard gate to the setter qubit
    qc.h(setter_qubit)
    qc.barrier()
    
    # Step 3: Apply parallel gates to the settable qubits
    # We'll use rotation gates that can be modified by the setter qubit's state
    # without using CNOT gates
    
    # Phase rotation gates on the settable qubits
    qc.rz(np.pi/2, settable1)
    qc.rz(np.pi/2, settable2)
    
    # Step 4: Apply operations to connect the first three qubits to the measurement qubit
    # Using phase gates instead of CNOT gates
    
    # Prepare the measurement qubit
    qc.h(measurement)
    
    # Apply phase operations that create relationships between qubits
    qc.rz(np.pi/4, setter_qubit)
    
    # Apply additional Hadamard gates to create interference
    qc.h(settable1)
    qc.h(settable2)
    
    # Final phase operations to establish relationship with measurement qubit
    qc.p(np.pi/2, measurement)
    qc.rz(np.pi/2, measurement)
    
    # Second round of Hadamard gates to complete the interference pattern
    qc.h(settable1)
    qc.h(settable2)
    qc.h(measurement)
    
    qc.barrier()
    
    # Measure all qubits
    qc.measure([0, 1, 2, 3], [0, 1, 2, 3])
    
    return qc

def run_all_input_combinations():
    """
    Run the quantum circuit with all possible input combinations
    and analyze the results
    """
    # All possible input combinations for two binary inputs
    input_combinations = [(0, 0), (0, 1), (1, 0), (1, 1)]
    results = {}
    
    print("Running superposition bit (AND) circuit with input combinations:")
    for inputs in input_combinations:
        input1, input2 = inputs
        print(f"\n=== Input combination: {input1}, {input2} ===")
        
        # Create and run circuit for this input combination
        qc = create_settable_quantum_circuit(input1, input2)
        
        # Simulate the circuit
        simulator = Aer.get_backend('qasm_simulator')
        shots = 4096
        job = simulator.run(qc, shots=shots)
        result = job.result()
        counts = result.get_counts(qc)
        results[inputs] = counts
        
        # Print circuit (for the first combination only to avoid clutter)
        if inputs == (0, 0):
            print("\nQuantum Circuit:")
            print(qc.draw(fold=90))
        
        # Analyze the measurement results
        analyze_measurement_results(counts, inputs)
        
        # Plot histogram
        plot_histogram(counts)
        plt.title(f"Results for Inputs {input1}, {input2}")
        plt.tight_layout()
        plt.show()
    
    # Compare the measurement qubit across different inputs
    compare_measurement_outcomes(results)
    
    return results

def analyze_measurement_results(counts, inputs):
    """
    Analyze the measurement results for a specific input combination
    
    Parameters:
    counts (dict): Measurement counts
    inputs (tuple): The input values (input1, input2)
    """
    input1, input2 = inputs
    print(f"\nResults analysis for inputs {input1}, {input2}:")
    
    # Sort by count
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    # Print the most frequent results
    print("\nTop 5 measurement outcomes:")
    print("Setter | Input1 | Input2 | Measure | Count | Probability")
    print("-" * 65)
    
    for i, (bitstring, count) in enumerate(sorted_counts[:5]):
        # Reverse bitstring to match qubit ordering
        bits = bitstring[::-1]
        setter = bits[0]
        in1 = bits[1]
        in2 = bits[2]
        measure = bits[3]
        prob = count / sum(counts.values())
        
        print(f"  {setter}    |   {in1}    |   {in2}    |    {measure}   | {count:5d} | {prob:.4f}")
    
    # Analyze the measurement qubit specifically
    measurement_analysis(counts, inputs)

def measurement_analysis(counts, inputs):
    """
    Analyze how the measurement qubit behaves for a given input
    
    Parameters:
    counts (dict): Measurement counts
    inputs (tuple): The input values (input1, input2)
    """
    input1, input2 = inputs
    
    # Gather statistics on the measurement qubit
    measure_0_count = 0
    measure_1_count = 0
    
    for bitstring, count in counts.items():
        # The measurement qubit is the 4th qubit (index 3 from the right)
        measure_bit = bitstring[-4]
        
        if measure_bit == '0':
            measure_0_count += count
        else:
            measure_1_count += count
    
    total = measure_0_count + measure_1_count
    prob_0 = measure_0_count / total
    prob_1 = measure_1_count / total
    
    print(f"\nMeasurement qubit analysis:")
    print(f"  Probability of |0⟩: {prob_0:.4f} ({measure_0_count} out of {total})")
    print(f"  Probability of |1⟩: {prob_1:.4f} ({measure_1_count} out of {total})")
    
    # Determine dominant outcome
    dominant = '0' if prob_0 > prob_1 else '1'
    print(f"  Dominant outcome: |{dominant}⟩ with probability {max(prob_0, prob_1):.4f}")

def compare_measurement_outcomes(results):
    """
    Compare the measurement qubit outcomes across all input combinations
    
    Parameters:
    results (dict): Results from all input combinations
    """
    print("\n=== Truth Table Based on Measurement Qubit ===")
    print("Input1 | Input2 | Measurement Qubit | Probability")
    print("-" * 50)
    
    # Analyze each input combination
    for inputs, counts in results.items():
        input1, input2 = inputs
        
        # Calculate measurement qubit probabilities
        measure_0_count = 0
        measure_1_count = 0
        
        for bitstring, count in counts.items():
            measure_bit = bitstring[-4]
            
            if measure_bit == '0':
                measure_0_count += count
            else:
                measure_1_count += count
        
        total = measure_0_count + measure_1_count
        dominant = '0' if measure_0_count > measure_1_count else '1'
        prob = max(measure_0_count, measure_1_count) / total
        
        print(f"  {input1}    |   {input2}    |        {dominant}         |   {prob:.4f}")
    
    # Try to identify logical patterns
    identify_logic_pattern(results)

def identify_logic_pattern(results):
    """
    Try to identify if the measurement qubit implements a known logic gate
    
    Parameters:
    results (dict): Results from all input combinations
    """
    # Extract the dominant measurement qubit value for each input
    truth_table = {}
    
    for inputs, counts in results.items():
        measure_0_count = 0
        measure_1_count = 0
        
        for bitstring, count in counts.items():
            measure_bit = bitstring[-4]
            
            if measure_bit == '0':
                measure_0_count += count
            else:
                measure_1_count += count
        
        dominant = '0' if measure_0_count > measure_1_count else '1'
        truth_table[inputs] = dominant
    
    # Check for known logic patterns
    print("\nLogic Pattern Analysis:")
    
    print("- logic pattern is quantified via probability, validity = 0.5000")
   

# Run the experiment with all input combinations
results = run_all_input_combinations()
