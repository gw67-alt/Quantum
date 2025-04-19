# Import necessary Qiskit libraries
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
# Use BasicSimulator directly if qiskit-aer is not the focus
# from qiskit_aer import AerSimulator
from qiskit.providers.basic_provider import BasicSimulator
from qiskit import transpile
import matplotlib.pyplot as plt # Optional: for drawing circuits
import math # Needed for sqrt if optimizing is_prime later, though not used here
import numpy as np # Needed for np.sign if used, though not used in this gate logic

# --- Configuration ---
# Simulator backend
# sim_backend = AerSimulator() # Use AerSimulator for more features/performance
sim_backend = BasicSimulator() # Use BasicSimulator for simplicity

# --- Helper Function ---
def run_gate_test(gate_name, circuit_builder_func, num_qubits, input_indices, output_index):
    """
    Tests a quantum circuit implementing a logic gate for all classical inputs.

    Args:
        gate_name (str): Name of the gate (for printing).
        circuit_builder_func (function): A function that takes a QuantumCircuit
                                         and QuantumRegister and adds the gate logic.
        num_qubits (int): Total number of qubits needed for the circuit.
        input_indices (list[int]): Indices of the input qubits.
        output_index (int): Index of the output qubit to measure.
    """
    print(f"\n--- Testing {gate_name} Gate ---")
    num_inputs = len(input_indices)
    num_classical_inputs = 2**num_inputs

    # Store expected results based on classical logic
    expected_results = {}
    if gate_name == "NOT":
        expected_results = { (0,): 1, (1,): 0 }
    elif gate_name == "XOR":
        expected_results = { (0,0): 0, (0,1): 1, (1,0): 1, (1,1): 0 }
    elif gate_name == "AND":
        expected_results = { (0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1 }
    elif gate_name == "OR":
        expected_results = { (0,0): 0, (0,1): 1, (1,0): 1, (1,1): 1 }


    for i in range(num_classical_inputs):
        # Determine input state (e.g., for 2 inputs: 0->00, 1->01, 2->10, 3->11)
        input_state_str = format(i, f'0{num_inputs}b')

        # Setup circuit for this input
        qr = QuantumRegister(num_qubits, 'q')
        cr = ClassicalRegister(1, 'c') # Measure only the output bit
        circuit = QuantumCircuit(qr, cr)

        # Initialize input qubits based on input_state_str
        current_input_vals = []
        input_tuple = [] # For dictionary lookup
        for idx, bit_char in enumerate(input_state_str):
            input_q_idx = input_indices[idx]
            bit_val = int(bit_char)
            if bit_val == 1:
                circuit.x(qr[input_q_idx])
            current_input_vals.append(bit_val)
            input_tuple.append(bit_val)

        circuit.barrier() # Separate initialization

        # Build the specific gate logic
        circuit_builder_func(circuit, qr)

        circuit.barrier() # Separate logic

        # Measure the output qubit
        circuit.measure(qr[output_index], cr[0])

        # Simulate
        # Transpilation is good practice, though less critical for BasicSimulator
        t_circuit = transpile(circuit, sim_backend)
        job = sim_backend.run(t_circuit, shots=100) # Fewer shots needed for deterministic gates
        result = job.result()
        counts = result.get_counts(t_circuit)

        # Determine measured output (most frequent result)
        # Handle cases where counts might be empty or have unexpected keys
        if counts:
             measured_output = max(counts, key=counts.get)
        else:
             measured_output = "N/A" # Or handle error appropriately

        # Get expected output
        expected_output = expected_results.get(tuple(input_tuple), "N/A")

        print(f"Input: {input_state_str} ({current_input_vals}) -> Output: {measured_output} (Expected: {expected_output})")

    # Optional: Draw the last circuit instance for visualization
    # print(f"\nCircuit diagram for {gate_name} (last input case):")
    # try:
    #     print(circuit.draw(output='text'))
    #     # circuit.draw(output='mpl', filename=f'{gate_name}_circuit.png') # Save plot
    # except ImportError:
    #     print("Install 'pylatexenc' for text drawing or 'matplotlib' for mpl drawing.")


# --- Gate Building Functions ---

# 1. NOT Gate
def build_not_circuit(qc, qr):
    """Applies NOT (X) gate to qr[0]."""
    # Apply X gate to the input qubit (index 0)
    qc.x(qr[0])

# 2. XOR Gate
def build_xor_circuit(qc, qr):
    """Implements XOR(qr[0], qr[1]) onto qr[2]."""
    # q2 = q0 XOR q1 (assuming q2 starts at |0>)
    qc.cx(qr[0], qr[2]) # CNOT(q0, q2)
    qc.cx(qr[1], qr[2]) # CNOT(q1, q2)

# 3. AND Gate
def build_and_circuit(qc, qr):
    """Implements AND(qr[0], qr[1]) onto qr[2]."""
    # q2 = q0 AND q1 (assuming q2 starts at |0>)
    qc.ccx(qr[0], qr[1], qr[2]) # Toffoli gate

# 4. OR Gate
def build_or_circuit(qc, qr):
    """Implements OR(qr[0], qr[1]) onto qr[2]."""
    # Implement q2 = q0 OR q1 using De Morgan's law
    # q2 = NOT ((NOT q0) AND (NOT q1))

    # Apply NOT to inputs
    qc.x(qr[0])
    qc.x(qr[1])

    # Apply AND (Toffoli) to the inverted inputs
    qc.ccx(qr[0], qr[1], qr[2]) # q2 = (NOT q0) AND (NOT q1)

    # Apply NOT to the result
    qc.x(qr[2]) # q2 = NOT ((NOT q0) AND (NOT q1)) = q0 OR q1

    # --- Important: Uncomputation ---
    # Return the input qubits to their original state.
    qc.x(qr[0])
    qc.x(qr[1])


# --- Run Tests ---
if __name__ == "__main__":
    # Test the NOT gate (1 input qubit, output is the same qubit)
    run_gate_test("NOT", build_not_circuit, num_qubits=1, input_indices=[0], output_index=0)

    # Test the XOR gate (2 input qubits, 1 output qubit)
    run_gate_test("XOR", build_xor_circuit, num_qubits=3, input_indices=[0, 1], output_index=2)

    # Test the AND gate (2 input qubits, 1 output qubit)
    run_gate_test("AND", build_and_circuit, num_qubits=3, input_indices=[0, 1], output_index=2)

    # Test the OR gate (2 input qubits, 1 output qubit)
    run_gate_test("OR", build_or_circuit, num_qubits=3, input_indices=[0, 1], output_index=2)
