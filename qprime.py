from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
import numpy as np
import math

def encode_number(n, num_qubits):
    """
    Encode a decimal number n into a quantum state
    using qubit rotations to represent the number in base 10
    """
    # Create a quantum circuit with specified number of qubits
    qr = QuantumRegister(num_qubits, 'q')
    cr = ClassicalRegister(num_qubits, 'c')
    circuit = QuantumCircuit(qr, cr)
    
    # Convert n to binary representation and pad with zeros
    binary = format(n, f'0{num_qubits}b')
    
    # Apply X gates where there are 1s in the binary representation
    for i, bit in enumerate(reversed(binary)):
        if bit == '1':
            circuit.x(qr[i])
    
    # Apply rotation gates to encode the number's magnitude
    angle = (n / (2**num_qubits - 1)) * np.pi
    circuit.ry(angle, qr[0])
    
    return circuit, qr, cr

def is_divisible_by(circuit, qr, cr, divisor, num_qubits):
    """
    Test if the encoded number is divisible by a given divisor
    Returns a circuit that marks the result in an ancilla qubit
    """
    # Add ancilla qubit for the result
    anc = QuantumRegister(1, 'anc')
    anc_c = ClassicalRegister(1, 'anc_c')
    
    # Create new circuit with the ancilla
    new_circ = QuantumCircuit(qr, anc, cr, anc_c)
    
    # Copy the previous circuit
    new_circ.compose(circuit, inplace=True)
    
    # Prepare the ancilla in |->
    new_circ.x(anc[0])
    new_circ.h(anc[0])
    
    # We'll use quantum modular arithmetic to check divisibility
    # For demonstration, we'll use a simplified approach
    # Real implementation would use quantum arithmetic circuits
    
    # Apply controlled phase rotation based on divisibility
    # Each qubit contributes 2^i to the value
    for i in range(num_qubits):
        remainder = (2**i) % divisor
        if remainder != 0:
            phase_angle = (2 * np.pi * remainder) / divisor
            new_circ.cp(phase_angle, qr[i], anc[0])
    
    # Apply H to convert phase kickback to bit value
    new_circ.h(anc[0])
    
    return new_circ, anc, anc_c

def check_prime(number, max_qubits=8):
    """
    Check if a number is prime using quantum circuit
    """
    if number <= 1:
        return False
    if number <= 3:
        return True
    if number % 2 == 0:
        return False
    
    # Determine number of qubits needed
    num_qubits = max(math.ceil(math.log2(number)), max_qubits)
    
    # Encode the number
    circuit, qr, cr = encode_number(number, num_qubits)
    
    # We'll check divisibility for odd numbers up to sqrt(number)
    sqrt_n = int(math.sqrt(number)) + 1
    
    # Initialize the simulator
    simulator = AerSimulator()
    
    # Check divisibility by each potential divisor
    for divisor in range(3, sqrt_n, 2):
        test_circ, anc, anc_c = is_divisible_by(circuit, qr, cr, divisor, num_qubits)
        
        # Measure the ancilla
        test_circ.measure(anc[0], anc_c[0])
        
        # Compile and run the circuit
        compiled_circuit = transpile(test_circ, simulator)
        result = simulator.run(compiled_circuit, shots=1024).result()
        counts = result.get_counts(test_circ)
        
        # If we measure '0' with high probability, the number is divisible by divisor
        if '0' in counts and counts['0'] > counts.get('1', 0):
            return False
    
    return True

def find_primes_up_to(n):
    """Find prime numbers up to n using the quantum circuit"""
    primes = []
    for i in range(2, n+1):
        if check_prime(i):
            primes.append(i)
    return primes

# Example usage
def main():
    # Test some numbers
    test_numbers = range(99)
    
    for num in test_numbers:
        is_prime = check_prime(num)
        print(f"{num} is{'prime' if is_prime else 'not prime'}")
   

if __name__ == "__main__":
    main()
