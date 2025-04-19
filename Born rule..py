

# =============================================================================
# Imports
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np
import traceback # For detailed error printing

# Qiskit Core
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.visualization import plot_histogram

# Qiskit Aer for local simulation
try:
    from qiskit_aer import Aer
except ImportError:
    print("Warning: qiskit-aer not found. Local simulation will not be available.")
    print("Install using: pip install qiskit-aer")
    Aer = None

# Qiskit IBM Runtime 
# Ensure you have the necessary version installed (likely >= 0.20 for these APIs)
# pip install qiskit-ibm-runtime --upgrade
try:
    from qiskit_ibm_runtime import (
        QiskitRuntimeService, 
        Sampler,                 # SamplerV2 primitive
        Session, 
        SamplerOptions           # Specific options class for SamplerV2
    )
except ImportError as e:
     print(f"Error importing qiskit_ibm_runtime components: {e}")
     print("Please ensure qiskit-ibm-runtime is installed correctly.")
     print("Install using: pip install qiskit-ibm-runtime")
     # Set placeholders to allow script structure but prevent runtime errors later if imports failed
     QiskitRuntimeService = None
     Sampler = None
     Session = None
     SamplerOptions = None

# One-time setup (run this once)
QiskitRuntimeService.save_account(
    token='',
    channel='ibm_quantum',  # Specify the channel
    instance='ibm-q/open/main',  # This is the default open instance
    overwrite=True
)
# =============================================================================
# Circuit Creation Function
# =============================================================================
def create_charge_detector_circuit(input1=0, input2=0):
    """
    Creates a quantum circuit simulating charge detection via phase differences.
    
    Qubits:
    q[0]: Control qubit (Hadamard applied)
    q[1]: Charge1 qubit (positive phase applied)
    q[2]: Charge2 qubit (negative phase applied)
    q[3]: Measurement qubit (interacts with others)
    
    Parameters:
    input1 (int): Initial state for Charge1 qubit (0 or 1)
    input2 (int): Initial state for Charge2 qubit (0 or 1)
    
    Returns:
    QuantumCircuit: The constructed quantum circuit.
    """
    qreg = QuantumRegister(4, 'q')
    # Ensure classical register name is 'c' as expected by V2 result access below
    creg = ClassicalRegister(4, 'c') 
    qc = QuantumCircuit(qreg, creg, name=f"ChargeDetect_Inputs_{input1}{input2}")
    
    control = qreg[0]
    charge1 = qreg[1]
    charge2 = qreg[2]
    measure_q = qreg[3] # Renamed to avoid conflict with measure method
    
    # Step 1: Initialize input states
    if input1 == 1: qc.x(charge1)
    if input2 == 1: qc.x(charge2)
    qc.barrier(label="Inputs")
    
    # Step 2: Control qubit superposition
    qc.h(control)
    qc.barrier(label="Control_H")
    
    # Step 3: Apply charge characteristics (phases)
    qc.h(charge1)
    qc.p(-np.pi/4, charge1)  # Positive phase for charge1
    qc.h(charge2)
    qc.p(np.pi/4, charge2) # Negative phase for charge2
    qc.barrier(label="Charges")
    
    # Step 4: Setup measurement qubit interactions
    qc.h(measure_q)
    qc.p(np.pi/4, control)   # Interaction with control
    qc.p(np.pi/4, measure_q) # Self-interaction/phase setup for measure
    qc.p(np.pi/8, charge1)   # Positive coupling to charge1
    qc.p(-np.pi/8, charge2)  # Negative coupling to charge2
    qc.barrier(label="Interaction")

    # Step 5: Final interference for measurement basis change
    qc.h(charge1)
    qc.h(charge2)
    qc.h(measure_q)
    qc.barrier(label="Measure_Basis")
    
    # Step 6: Measure all qubits
    qc.measure(qreg, creg)
    
    return qc

# =============================================================================
# Results Analysis Function
# =============================================================================
def analyze_results(counts, inputs):
    """
    Analyzes and prints measurement results for a specific input combination.
    
    Parameters:
    counts (dict): Measurement counts {bitstring: count}.
    inputs (tuple): The input values (input1, input2).
    """
    input1, input2 = inputs
    print(f"\nResults analysis for inputs {input1}, {input2}:")
    
    if not counts:
        print("No counts received.")
        return
        
    print("\nRaw measurement outcomes:")
    # Qubit order: q3=measure, q2=charge2, q1=charge1, q0=control
    # Bitstring order: c3 c2 c1 c0 (Qiskit default)
    print("Bitstring | Meas | Chg2 | Chg1 | Ctrl | Count | Probability")
    print("-" * 77)
    
    # Sort by bitstring value for consistent presentation
    try:
        # Convert keys to int for sorting, assuming they represent binary numbers
        bit_ordered_counts = sorted(counts.items(), key=lambda item: int(str(item[0]), 2))
    except ValueError:
         print("Warning: Could not sort counts numerically, using string sort.")
         bit_ordered_counts = sorted(counts.items())

    
    total_shots = sum(counts.values())
    if total_shots == 0: 
        print("Total shots are zero.")
        return
        
    for bitstring, count in bit_ordered_counts:
        # Ensure bitstring is 4 chars long, padding if necessary
        # Convert int bitstring key back to binary string for display
        bitstring_str = format(bitstring, '04b') if isinstance(bitstring, int) else str(bitstring).zfill(4)

        # Extract bits according to standard Qiskit order (c[N-1]...c[0])
        measure_c = bitstring_str[0] # c3
        ch2_c = bitstring_str[1]     # c2
        ch1_c = bitstring_str[2]     # c1
        control_c = bitstring_str[3] # c0
        prob = count / total_shots
        
        print(f"{bitstring_str:^9s} | {measure_c:^4s} | {ch2_c:^4s} | {ch1_c:^4s} | {control_c:^4s} | {count:5d} | {prob:.4f}")

# =============================================================================
# IBM Runtime Execution Function (Includes Transpilation & V2 Result Handling)
# =============================================================================
def run_circuit_ibm_runtime(use_simulator=True, backend_name=None):
    """
    Runs the quantum circuit using IBM Runtime (SamplerV2 API), including transpilation.
    Handles backend selection, Session management, and Sampler execution.
    Uses updated result handling for PrimitiveResult.
    """
    # Check if runtime components were imported successfully
    if not QiskitRuntimeService or not Sampler or not Session or not SamplerOptions:
        print("Error: Qiskit Runtime components not available. Cannot run on IBM Quantum.")
        return {}
        
    print("\n--- Running on IBM Quantum ---")
    try:
        # Initialize service (loads saved credentials)
        service = QiskitRuntimeService() 
        print(f"Initialized QiskitRuntimeService (Account: {service.active_account()})")
    except Exception as e:
        print(f"Failed to initialize QiskitRuntimeService: {e}")
        print("Please ensure you have saved your account token (see main block).")
        return {}

    # --- Backend Selection ---
    actual_backend_name = None # Stores name for messages
    backend_object = None      # Stores the object needed for Session and transpile

    print("Selecting backend...")
    try:
        if use_simulator:
            sim_backend_name = "ibmq_qasm_simulator" 
            print(f"Attempting to get simulator backend: {sim_backend_name}")
            backend_object = service.backend(sim_backend_name) 
            actual_backend_name = backend_object.name 
            print(f"Using simulator: {actual_backend_name}")
        else:
            if backend_name is None:
                print("Finding least busy operational hardware backend (min 4 qubits)...")
                backends = service.backends(operational=True, simulator=False, min_num_qubits=4)
                if not backends:
                    raise RuntimeError("No suitable operational hardware backends found.") 
                least_busy_backend = min(backends, key=lambda b: b.status().pending_jobs)
                backend_object = least_busy_backend 
                actual_backend_name = backend_object.name 
                print(f"Selected least busy backend: {actual_backend_name}")
            else:
                print(f"Attempting to get specified hardware backend: {backend_name}")
                backend_object = service.backend(backend_name) 
                if not backend_object.operational:
                     raise RuntimeError(f"Specified backend '{backend_name}' is not operational.")
                actual_backend_name = backend_object.name
                print(f"Using specified backend: {actual_backend_name}")

    except Exception as e:
        print(f"Error during backend selection: {e}")
        print("Please check backend name, availability, and your account permissions.")
        traceback.print_exc()
        return {}

    if backend_object is None:
         print("Failed to obtain a valid backend object.")
         return {}
    # --- End Backend Selection ---

    input_combinations = [(0, 0), (0, 1), (1, 0), (1, 1)]
    results = {}
    
    print(f"\nRunning circuits on backend: {actual_backend_name}")

    # --- Options Setup (Using SamplerOptions for SamplerV2) ---
    options = SamplerOptions() 
    try:
        options.environment.log_level = "WARNING" 
        print("Set options.environment.log_level = 'WARNING'")
    except AttributeError:
        try:
            options.log_level = "WARNING" 
            print("Set options.log_level = 'WARNING'")
        except AttributeError as e:
            print(f"Note: Could not set log_level on SamplerOptions object ({e}).")
    # --- End Options Setup ---

    # --- Session and Sampler Execution ---
    session = None # Define session variable outside 'with' for finally block reference
    try:
        print(f"Attempting to start session with backend object: {actual_backend_name} (Type: {type(backend_object)})")
        
        with Session(backend=backend_object) as session: 
            print(f"Session active (ID: {session.session_id})") 

            sampler = Sampler(options=options) 
            print(f"SamplerV2 initialized within session.")

            num_shots = 8192 

            for inputs in input_combinations:
                input1, input2 = inputs
                print(f"\n=== Input combination: {input1}, {input2} ===")
                
                # 1. Create the ideal circuit
                qc = create_charge_detector_circuit(input1, input2)
                
                if inputs == (0, 0):
                    print("\nOriginal Quantum Circuit (Input 0,0):")
                    try:
                        print(qc.draw(fold=100)) 
                    except ImportError:
                         print("(Circuit drawing requires pylatexenc)")
                         print(qc) 

                # 2. Explicitly Transpile the circuit for the target backend
                print(f"Transpiling circuit for backend {actual_backend_name}...")
                transpiled_qc = None 
                try:
                    transpiled_qc = transpile(qc, backend=backend_object, optimization_level=1)
                    print("Transpilation complete.")
                except Exception as transpile_error:
                    print(f"!!! Error during transpilation: {transpile_error}")
                    print("Skipping execution for this input combination.")
                    traceback.print_exc() 
                    continue 

                # 3. Run the *transpiled* circuit using SamplerV2
                print(f"Submitting transpiled job for inputs {inputs} with {num_shots} shots...")
                job = sampler.run([transpiled_qc], shots=num_shots) 
                
                print(f"Job submitted (ID: {job.job_id()}). Waiting for results...")
                result = job.result() # This is a PrimitiveResult object
                print("Job completed.")

                # 4. Process results (V2 PrimitiveResult structure)
                # *** FIX 1 Applied Here: Access results via indexing and data attribute ***
                if len(result) == 0:
                     print("Warning: No results found in the PrimitiveResult.")
                     continue
                
                pub_result = result[0] # Get PubResult for the first circuit (index 0)

                # Check if the expected data structure exists
                if not hasattr(pub_result, 'data') or not hasattr(pub_result.data, 'c'):
                     print(f"Warning: Result structure unexpected for inputs {inputs}. Cannot find result[0].data.c.")
                     print(f"Available data fields: {pub_result.data}" if hasattr(pub_result, 'data') else "No data attribute")
                     continue
                     
                data_bin = pub_result.data.c # Access DataBin for classical register 'c'
                
                # Check if DataBin has probability method
                if not hasattr(data_bin, 'get_probabilities'):
                     print(f"Warning: DataBin object does not have 'get_probabilities' method for inputs {inputs}.")
                     # Attempt to get counts as a fallback
                     if hasattr(data_bin, 'get_counts'):
                          print("Attempting to use get_counts() instead.")
                          approx_counts = data_bin.get_counts()
                     else:
                          print("Cannot extract probabilities or counts. Skipping analysis for this input.")
                          continue
                else:
                     # Get probabilities and convert to approximate counts
                     prob_dict = data_bin.get_probabilities() 
                     approx_counts = {
                         # Convert binary string keys from prob_dict to int for consistency if needed
                         # Or keep as strings if analyze_results handles them
                         bitstring: max(0, int(round(prob * num_shots))) 
                         for bitstring, prob in prob_dict.items()
                     }
                # *** End of FIX 1 ***

                results[inputs] = approx_counts
                analyze_results(approx_counts, inputs) 
                
                # 5. Plot histogram
                if approx_counts: 
                    plot_histogram(approx_counts, 
                                   sort='value', 
                                   title=f"Results Inputs {input1},{input2} ({actual_backend_name})")
                    plt.tight_layout()
                    plt.show()
                else:
                    print("No measurement results to plot.")

    except TypeError as te:
        print(f"\nAn API mismatch error occurred (TypeError): {te}")
        print("Check constructor arguments for Session/Sampler and options types for your qiskit-ibm-runtime version.")
        traceback.print_exc()
        return {}
    except ValueError as ve:
         print(f"\nAn error occurred during initialization/execution (ValueError): {ve}")
         traceback.print_exc()
         return {}
    except Exception as e:
        print(f"\nAn unexpected error occurred during runtime execution: {e}")
        traceback.print_exc() 
        return {}
    finally:
        # *** FIX 2 Applied Here: Removed check for session.active ***
        # The 'with' context manager handles session closing automatically.
        # No need to check session.active here.
        print("--- Session block finished ---")
        # You could add other cleanup here if needed

    print("--- IBM Quantum Run Finished ---")
    return results
    # --- End Session and Sampler Execution ---

# =============================================================================
# Local Simulation Function (No transpilation needed typically)
# =============================================================================
def simulate_locally():
    """
    Runs the circuit using the local qasm_simulator from qiskit-aer.
    """
    print("\n--- Running Locally using qiskit-aer ---")
    if Aer is None:
        print("Error: qiskit-aer is not installed or failed to import. Cannot simulate locally.")
        return {}
        
    try:
        simulator = Aer.get_backend('qasm_simulator')
    except Exception as e:
        print(f"Could not get Aer qasm_simulator: {e}")
        print("Please ensure qiskit-aer is installed ('pip install qiskit-aer')")
        return {}

    input_combinations = [(0, 0), (0, 1), (1, 0), (1, 1)]
    results = {}
    
    for inputs in input_combinations:
        input1, input2 = inputs
        print(f"\n=== Input combination: {input1}, {input2} ===")
        
        # Create the ideal circuit (no transpilation needed for Aer simulator)
        qc = create_charge_detector_circuit(input1, input2)
        
        if inputs == (0, 0):
             print("\nQuantum Circuit (Input 0,0):")
             try:
                 print(qc.draw(fold=100))
             except ImportError:
                 print("(Circuit drawing requires pylatexenc)")
                 print(qc) 
        
        shots = 8192 
        print(f"Simulating with {shots} shots...")
        job = simulator.run(qc, shots=shots)
        result = job.result()
        counts = result.get_counts(qc)
        print("Simulation complete.")
        results[inputs] = counts
        
        analyze_results(counts, inputs)
        
        if counts:
            plot_histogram(counts, 
                           sort='value', 
                           title=f"Local Simulation Results Inputs {input1},{input2}") 
            plt.tight_layout()
            plt.show()
        else:
            print("No measurement results to plot.")
            
    print("--- Local Simulation Finished ---")
    return results

# =============================================================================
# Main Execution Block
# =============================================================================
if __name__ == "__main__":
    
    # --- IBM Quantum Account Setup (Run Once If Needed) ---
    # Uncomment and fill in your token if you haven't saved credentials.
    # try:
    #     if QiskitRuntimeService: 
    #         QiskitRuntimeService.save_account(
    #             token='YOUR_API_TOKEN', 
    #             channel='ibm_quantum',         
    #             instance='ibm-q/open/main',    
    #             overwrite=True
    #         )
    #         print("IBM Quantum account details potentially saved/updated.")
    #     else:
    #          print("Skipping account save: QiskitRuntimeService not available.")
    # except Exception as e:
    #      print(f"Error saving account: {e}")
    # --- End Account Setup ---

    print("Quantum Charge Detector Simulation")
    print("=" * 30)
    
    run_option = input(
        "Choose run option:\n"
        "  1. Local Simulator (qiskit-aer)\n"
        "  2. IBM Quantum Simulator (requires saved account, uses transpilation)\n"
        "  3. IBM Quantum Hardware (Least Busy, requires saved account, uses transpilation)\n"
        "  4. IBM Quantum Hardware (Specify Name, requires saved account, uses transpilation)\n"
        "Enter option number (1-4): "
    )

    results_data = {} 
    
    if run_option == '1':
        results_data = simulate_locally()
    elif run_option == '2':
        if QiskitRuntimeService:
             results_data = run_circuit_ibm_runtime(use_simulator=True)
        else:
             print("Cannot run on IBM Quantum: QiskitRuntimeService not available.")
    elif run_option == '3':
         if QiskitRuntimeService:
             results_data = run_circuit_ibm_runtime(use_simulator=False, backend_name=None)
         else:
              print("Cannot run on IBM Quantum: QiskitRuntimeService not available.")
    elif run_option == '4':
        if QiskitRuntimeService:
            backend_input = input("Enter the specific IBM Quantum backend name (e.g., ibm_brisbane): ")
            results_data = run_circuit_ibm_runtime(use_simulator=False, backend_name=backend_input)
        else:
             print("Cannot run on IBM Quantum: QiskitRuntimeService not available.")
    else:
        print("Invalid option selected. Exiting.")

    # Final summary message
    if results_data: 
        print("\n--- Experiment Summary ---")
        print("- Circuit simulates charge interaction via phases.")
        print("- Measurement outcomes depend on whether input charges (q1, q2) are same or different.")
        print("\nExecution finished.")
    else:
        print("\nExecution finished, but no results were generated (check for errors above).")

