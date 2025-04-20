# =============================================================================
# Imports
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np
import traceback # For detailed error printing
import datetime # For timestamp

# Qiskit Core
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
# Import MCXGate for multi-controlled operations (like Toffoli/CCX)
from qiskit.circuit.library import MCXGate # CCXGate is also available directly
from qiskit.visualization import plot_histogram

# Qiskit Aer for local simulation
try:
    from qiskit_aer import Aer
except ImportError:
    print("Warning: qiskit-aer not found. Local simulation will not be available.")
    print("Install using: pip install qiskit-aer")
    Aer = None

# Qiskit IBM Runtime
# Ensure qiskit-ibm-runtime is installed (pip install qiskit-ibm-runtime)
try:
    from qiskit_ibm_runtime import (
        QiskitRuntimeService,
        Sampler,          # SamplerV2 primitive
        Session,
        SamplerOptions    # Specific options class for SamplerV2 (though may not be used directly below)
    )
except ImportError as e:
      print(f"Error importing qiskit_ibm_runtime components: {e}")
      # Set placeholders
      QiskitRuntimeService = None
      Sampler = None
      Session = None
      SamplerOptions = None

# =============================================================================
# IBM Account Setup (Run Once if Needed)
# =============================================================================
def setup_ibm_account(token='', instance='ibm-q/open/main'):
    """Saves IBM Quantum account credentials."""
    if not QiskitRuntimeService:
        print("QiskitRuntimeService not available. Cannot save account.")
        return False
    try:
        QiskitRuntimeService.save_account(
            token=token,
            channel='ibm_quantum',  # Specify the channel
            instance=instance,      # Default open instance or specify yours
            overwrite=True
        )
        print(f"Account saved for instance '{instance}'.")
        return True
    except Exception as e:
        print(f"Error saving IBM Quantum account: {e}")
        print("Please ensure you have a valid API token.")
        return False

# =============================================================================
# Circuit Creation Function (Binary Incrementer - CORRECTED LOGIC)
# =============================================================================
def create_binary_incrementer_circuit(num_qubits=3, initial_state_str="000"):
    """
    Creates a quantum circuit that increments a binary number represented by qubits.
    Increments |x> to |x+1 mod 2^n>. Uses X, CX, CCX (MCX) gates.
    *** Uses corrected ripple-carry increment logic. ***

    Qubits: q[0] is the least significant bit (LSB), q[n-1] is the MSB.

    Parameters:
    num_qubits (int): The number of qubits for the counter.
    initial_state_str (str): Binary string representing the initial state (e.g., "011").

    Returns:
    QuantumCircuit: The constructed incrementer circuit.
    """
    if len(initial_state_str) != num_qubits or not all(bit in '01' for bit in initial_state_str):
        raise ValueError(f"initial_state_str must be a {num_qubits}-bit binary string.")

    qreg = QuantumRegister(num_qubits, 'q')
    creg = ClassicalRegister(num_qubits, 'c') # Measure all qubits
    qc = QuantumCircuit(qreg, creg, name=f"Inc_{initial_state_str}")

    # Step 1: Initialize Input State (LSB q[0] corresponds to rightmost bit)
    # Reverse the string because qiskit's LSB is q[0]
    for i, bit in enumerate(reversed(initial_state_str)):
        if bit == '1':
            qc.x(qreg[i])
    qc.barrier(label="Initial State")

    # Step 2: Apply CORRECTED Increment Logic (Ripple-Carry Adder adding 1)
    # Apply multi-controlled gates first, from highest bit down to CX
    for i in range(num_qubits - 1, 0, -1): # Loop from n-1 down to 1
        control_qubits = [qreg[j] for j in range(i)] # Controls q[0]...q[i-1]
        target_qubit = qreg[i]
        qc.mcx(control_qubits, target_qubit)

    # Flip the LSB (q[0]) last
    if num_qubits > 0:
        qc.x(qreg[0])

    qc.barrier(label="Increment")

    # Step 3: Measure all qubits
    # Measure q[i] -> c[i]
    qc.measure(qreg, creg)

    return qc

# =============================================================================
# Results Analysis Function (for Binary Incrementer Circuit)
# =============================================================================
def analyze_incrementer_results(counts, num_qubits, initial_state_str):
    """
    Analyzes measurement results for the n-bit output of the incrementer circuit.
    Checks if the output matches the expected incremented state.

    Parameters:
    counts (dict): Measurement counts {'c(n-1)...c0': count, ...}.
    num_qubits (int): The number of qubits.
    initial_state_str (str): The binary string of the initial state.
    """
    print(f"\nResults analysis for initial state |{initial_state_str}>:")

    if not counts: print("No counts received."); return

    # Expected result
    initial_state_int = int(initial_state_str, 2)
    expected_final_int = (initial_state_int + 1) % (2**num_qubits)
    expected_outcome_str = format(expected_final_int, f'0{num_qubits}b')

    print("\nMeasured outcomes:")
    # Header depends on num_qubits, create dynamically
    # Qiskit bitstring is c(n-1)...c0 corresponding to q(n-1)...q[0]
    header = "Outcome | " + " | ".join([f"q{i}(c{i})" for i in reversed(range(num_qubits))]) + " | Count | Probability | Expected"
    print(header)
    print("-" * len(header))

    normalized_counts = {}
    total_shots = 0
    valid_keys = [format(i, f'0{num_qubits}b') for i in range(2**num_qubits)]

    for key, count in counts.items():
        total_shots += count
        # Ensure key is a valid n-bit string
        key_str = format(key, f'0{num_qubits}b') if isinstance(key, int) else str(key)
        if len(key_str) == num_qubits and all(bit in '01' for bit in key_str):
             normalized_counts[key_str] = normalized_counts.get(key_str, 0) + count
        else:
             print(f"Warning: Ignoring invalid result key '{key}'.")

    if total_shots == 0: print("Total valid shots zero."); return

    # Ensure all possible outcomes are represented for printing
    for key in valid_keys:
        if key not in normalized_counts:
            normalized_counts[key] = 0

    # Print results for each possible outcome
    for outcome in sorted(normalized_counts.keys()):
        count = normalized_counts[outcome]
        prob = count / total_shots if total_shots > 0 else 0
        # Display bits corresponding to q[n-1] ... q[0]
        bits_display = " | ".join(list(outcome)) # qiskit bitstring is c(n-1)..c0
        is_expected = "<- Expected" if outcome == expected_outcome_str else ""
        print(f"{outcome:^7s} | {bits_display} | {count:5d} | {prob:.4f}      | {is_expected}")

    # Interpretation
    if total_shots > 0:
        # Find the outcome with the highest probability
        # Handle potential empty counts dict after filtering
        if not normalized_counts:
             print("Interpretation: No valid counts after normalization.")
             return

        majority_outcome = max(normalized_counts, key=normalized_counts.get)
        # Check if majority outcome exists and has non-zero count
        if normalized_counts[majority_outcome] == 0:
             print(f"Interpretation: No counts observed. Expected |{expected_outcome_str}>.")
        elif normalized_counts[majority_outcome] / total_shots > 0.7: # Higher confidence for counters
            if majority_outcome == expected_outcome_str:
                print(f"Interpretation: Correctly measured final state |{expected_outcome_str}>.")
            else:
                print(f"Interpretation: INCORRECT majority outcome |{majority_outcome}> (expected |{expected_outcome_str}>). Noise likely.")
        # Check if results are mixed (e.g., close to 50/50 for the expected state)
        elif abs(normalized_counts.get(expected_outcome_str, 0)/total_shots - 0.5) < 0.2 and len(normalized_counts) > 1 :
             print(f"Interpretation: Results are mixed. Expected |{expected_outcome_str}>. Potential issue or high noise.")
        else: # If no clear majority and not clearly mixed around expected
             print(f"Interpretation: No clear majority outcome. Expected |{expected_outcome_str}>. Observed: {majority_outcome} (Prob: {normalized_counts[majority_outcome]/total_shots:.3f})")
    else: print("Interpretation: No valid counts.")
    print(f"(Note: Circuit attempts to increment |{initial_state_str}> to |{expected_outcome_str}>)")


# =============================================================================
# IBM Runtime Execution Function (MODIFIED FOR Incrementer)
# =============================================================================
def run_incrementer_circuit_ibm_runtime(num_qubits, initial_states, use_simulator=True, backend_name=None):
    """
    MODIFIED: Runs the Binary Incrementer circuit for given initial states using IBM Runtime.
    Calls create_binary_incrementer_circuit and analyze_incrementer_results.
    """
    if not QiskitRuntimeService or not Sampler or not Session:
        print("IBM Runtime components not available.")
        return {}
    print("\n--- Running Binary Incrementer circuit on IBM Quantum ---")
    try:
        # Ensure account is loaded or provide token/instance details
        service = QiskitRuntimeService()
        print(f"Initialized QiskitRuntimeService (Account: {service.active_account()})")
    except Exception as e:
        print(f"Failed init QiskitRuntimeService: {e}")
        print("Please ensure your account is saved (e.g., using setup_ibm_account).")
        return {}


    # --- Backend Selection ---
    actual_backend_name = None; backend_object = None; min_qubits_needed = num_qubits
    print("Selecting backend...")
    try:
        if use_simulator:
            # Use a simulator available through the runtime service
            sim_backend_name = "ibmq_qasm_simulator" # Common simulator name
            backend_object = service.backend(sim_backend_name)
        else: # Hardware
            if backend_name is None: # Find least busy suitable hardware
                backends = service.backends(operational=True, simulator=False, min_num_qubits=min_qubits_needed)
                if not backends: raise RuntimeError(f"No suitable operational hardware ({min_qubits_needed}+ qubits) found.")
                # Filter further? e.g., by architecture if needed
                def get_pending_jobs(b):
                    try: return b.status().pending_jobs
                    except: return float('inf') # Prioritize backends where status can be read
                backend_object = min(backends, key=get_pending_jobs)
            else: # Use specified name
                backend_object = service.backend(backend_name)
                if backend_object.num_qubits < min_qubits_needed: raise RuntimeError(f"Backend '{backend_name}' needs >= {min_qubits_needed} qubits.")
                if not backend_object.operational: raise RuntimeError(f"Backend '{backend_name}' not operational.")
        actual_backend_name = backend_object.name
        print(f"Using backend: {actual_backend_name}")
    except Exception as e: print(f"Backend selection error: {e}"); traceback.print_exc(); return {}
    if backend_object is None: print("Failed getting backend object."); return {}
    # --- End Backend Selection ---

    results = {}
    print(f"\nRunning circuits on backend: {actual_backend_name}")

    # --- Options Setup ---
    print("Relying on default Sampler resilience settings (check documentation for levels).")
    # If specific options needed: options = SamplerOptions(optimization_level=1, resilience_level=1)
    transpile_optimization_level = 1 # Set transpilation optimization level
    # --- End Options Setup ---

    # --- Session and Sampler Execution ---
    try:
        print(f"Attempting session with backend: {actual_backend_name}")
        # Pass backend name or object to Session
        with Session(service=service, backend=actual_backend_name) as session:
            print(f"Session active (ID: {session.session_id})")
            # Pass session to Sampler explicitly if needed by version, otherwise relies on context
            sampler = Sampler(session=session) # Pass session explicitly
            print(f"SamplerV2 initialized (using active session).")
            num_shots = 8192 # Standard shots

            circuits_to_run = []; circuit_inputs = []
            print("\nPreparing and transpiling circuits...")
            print(f"Using transpilation optimization_level={transpile_optimization_level}")

            for initial_state_str in initial_states:
                # *** CALL THE INCREMENTER CIRCUIT FUNCTION ***
                qc = create_binary_incrementer_circuit(num_qubits, initial_state_str)
                try:
                    # Pass backend object to transpile
                    transpiled_qc = transpile(qc, backend=backend_object,
                                              optimization_level=transpile_optimization_level)
                    circuits_to_run.append(transpiled_qc)
                    circuit_inputs.append(initial_state_str) # Track input state
                    print(f"  - Circuit for initial state |{initial_state_str}> prepared and transpiled.")
                except Exception as transpile_error:
                    print(f"  !!! Error transpiling initial state |{initial_state_str}>: {transpile_error}")
                    traceback.print_exc()


            if not circuits_to_run:
                print("No circuits were successfully transpiled.")
                return {}

            print(f"\nSubmitting batch job ({len(circuits_to_run)} circuits, {num_shots} shots each)...")
            # Pass list of circuits to sampler
            job = sampler.run(circuits=circuits_to_run, shots=num_shots)
            print(f"Job submitted (ID: {job.job_id()}). Waiting for results...")
            primitive_result = job.result() # This retrieves the results object
            print("Job completed.")

            print("\nProcessing results...")
            # The result object contains metadata and results for each circuit (pub)
            if not hasattr(primitive_result, 'pubs'):
                 print("Error: Primitive result does not contain 'pubs'.")
                 print(f"Received result type: {type(primitive_result)}")
                 return {}

            for i, pub_result in enumerate(primitive_result.pubs):
                 if i >= len(circuit_inputs): continue # Safety check
                 current_initial_state = circuit_inputs[i]

                 # Check structure for SamplerV2 results within each pub
                 if not hasattr(pub_result, 'data') or not hasattr(pub_result.data, 'c'):
                      print(f"Warning: Unexpected result structure in pub {i} for initial state |{current_initial_state}>. Skipping.")
                      print(f"Received pub_result: {pub_result}")
                      continue

                 # Access counts for the classical register 'c'
                 data_bin = pub_result.data.c # Access the data container for register 'c'
                 run_counts = data_bin.get_counts() # Get counts from the data container

                 print(f"\n--- Extracted counts for Initial State |{current_initial_state}> ---")
                 results[current_initial_state] = run_counts
                 # *** CALL THE INCREMENTER ANALYSIS FUNCTION ***
                 analyze_incrementer_results(run_counts, num_qubits, current_initial_state)

                 if run_counts:
                     # Ensure keys are strings for plotting
                     plot_counts = {format(k, f'0{num_qubits}b') if isinstance(k, int) else str(k): v for k, v in run_counts.items()}
                     plot_histogram(plot_counts, sort='value',
                                    title=f"Incrementer Results Init |{current_initial_state}> ({actual_backend_name})")
                     plt.tight_layout(); plt.show(block=False) # Non-blocking plot display

    # Error Handling
    except TypeError as te: print(f"\nAPI usage error (TypeError): {te}"); traceback.print_exc(); return {}
    except ValueError as ve: print(f"\nValue error: {ve}"); traceback.print_exc(); return {}
    except Exception as e: print(f"\nUnexpected error during runtime execution: {e}"); traceback.print_exc(); return {}
    finally:
        # Ensure all plots are displayed at the very end if execution reaches here
        if results:
            print("\nFinalizing plots...")
            plt.show() # Blocking call to show all figures
        print("--- Session block finished ---")


    print("\n--- IBM Quantum Run Finished (Binary Incrementer) ---")
    return results

# =============================================================================
# Local Simulation Function (MODIFIED FOR Incrementer)
# =============================================================================
def simulate_incrementer_circuit_locally(num_qubits, initial_states):
    """
    MODIFIED: Runs the Binary Incrementer circuit using qiskit-aer.
    Calls create_binary_incrementer_circuit and analyze_incrementer_results.
    """
    print("\n--- Running Binary Incrementer circuit Locally using qiskit-aer ---")
    if Aer is None: print("qiskit-aer not found."); return {}
    try: simulator = Aer.get_backend('qasm_simulator')
    except Exception as e: print(f"Could not get qasm_simulator: {e}"); return {}

    results = {}
    print("Preparing and simulating circuits locally...")
    for initial_state_str in initial_states:
        # *** CALL THE INCREMENTER CIRCUIT FUNCTION ***
        qc = create_binary_incrementer_circuit(num_qubits, initial_state_str)
        shots = 8192
        print(f"  - Simulating circuit for initial state |{initial_state_str}> with {shots} shots...")
        # Transpile for simulator optimization (optional but good practice)
        try:
            compiled_circuit = transpile(qc, simulator)
            job = simulator.run(compiled_circuit, shots=shots)
            result = job.result()
            counts = result.get_counts(compiled_circuit) # Get counts using the compiled circuit
            print("  - Simulation complete.")
            results[initial_state_str] = counts
            # *** CALL THE INCREMENTER ANALYSIS FUNCTION ***
            analyze_incrementer_results(counts, num_qubits, initial_state_str)
            if counts:
                plot_counts = {format(k, f'0{num_qubits}b') if isinstance(k, int) else str(k): v for k, v in counts.items()}
                plot_histogram(plot_counts, sort='value',
                               title=f"Local Sim Incrementer Results Init |{initial_state_str}>")
                plt.tight_layout(); plt.show(block=False) # Non-blocking plot
        except Exception as sim_error:
            print(f"  !!! Error during local simulation for |{initial_state_str}>: {sim_error}")
            traceback.print_exc()


    if results:
        print("\nFinalizing plots...")
        plt.show() # Show all plots at the end
    print("\n--- Local Simulation Finished (Binary Incrementer) ---")
    return results

# =============================================================================
# Main Execution Block (MODIFIED FOR Binary Incrementer)
# =============================================================================
if __name__ == "__main__":

    # --- Optional: Setup IBM Account ---
    # If needed, uncomment and run once with your token.
    # print("Please enter your IBM Quantum API token (leave blank if already saved):")
    # token = input()
    # if token: setup_ibm_account(token=token)
    # print("-" * 60)
    # --- End Optional Setup ---


    print("Quantum Binary Counter (Incrementer) Simulation")
    print("=" * 60)
    now_local = datetime.datetime.now(datetime.timezone.utc).astimezone()
    print(f"Current Time: {now_local.strftime('%Y-%m-%d %H:%M:%S %Z%z')}")

    # --- Configuration ---
    NUM_COUNTER_QUBITS = 3 # Set the size of the counter
    # Define initial states to test (must match NUM_COUNTER_QUBITS)
    # Test all possible states for the given number of qubits
    initial_states_to_test = [format(i, f'0{NUM_COUNTER_QUBITS}b') for i in range(2**NUM_COUNTER_QUBITS)]
    # Or test specific states:
    # initial_states_to_test = ["000", "011", "110", "111"]

    # Ensure states list is not empty if custom list used
    if not initial_states_to_test:
         initial_states_to_test.append(format(0, f'0{NUM_COUNTER_QUBITS}b')) # Add default if list empty

    print(f"(Simulating a {NUM_COUNTER_QUBITS}-qubit binary incrementer: |x> -> |x+1 mod {2**NUM_COUNTER_QUBITS}>)")
    print(f"(Testing initial states: {', '.join(initial_states_to_test)})")
    print("-" * 60)

    # Draw an example circuit
    example_state = initial_states_to_test[0] # Draw for the first state in the list
    qc_example = create_binary_incrementer_circuit(NUM_COUNTER_QUBITS, initial_state_str=example_state)
    try:
        print(f"Circuit diagram for initial state |{example_state}>:")
        # Increase fold width if needed for larger circuits
        print(qc_example.draw(output='text', fold=120))
    except ImportError:
        print("Circuit drawing requires pylatexenc. Using basic print(qc):")
        print(qc_example)
    except Exception as e:
         print(f"An error occurred during drawing: {e}")
         print(qc_example) # Fallback
    print("-" * 60)

    # Choose execution backend
    min_req_qubits = NUM_COUNTER_QUBITS
    run_option = input(
        "Choose run option:\n"
        f"  1. Local Simulator (qiskit-aer, faster, ideal)\n"
        f"  2. IBM Quantum Simulator (cloud, >= {min_req_qubits} qubits)\n"
        f"  3. IBM Quantum Hardware (Least Busy >= {min_req_qubits} qubits, real device)\n"
        f"  4. IBM Quantum Hardware (Specify Name >= {min_req_qubits} qubits, real device)\n"
        "Enter option number (1-4): "
    )

    results_data = {} # Will store the counts dictionary for each initial state

    if run_option == '1':
        results_data = simulate_incrementer_circuit_locally(NUM_COUNTER_QUBITS, initial_states_to_test)
    elif run_option == '2':
        if QiskitRuntimeService:
             results_data = run_incrementer_circuit_ibm_runtime(NUM_COUNTER_QUBITS, initial_states_to_test, use_simulator=True)
        else: print("Cannot run on IBM Quantum: QiskitRuntimeService not available.")
    elif run_option == '3':
         if QiskitRuntimeService:
             results_data = run_incrementer_circuit_ibm_runtime(NUM_COUNTER_QUBITS, initial_states_to_test, use_simulator=False, backend_name=None)
         else: print("Cannot run on IBM Quantum: QiskitRuntimeService not available.")
    elif run_option == '4':
        if QiskitRuntimeService:
            backend_input = input(f"Enter the specific IBM Quantum backend name (>= {min_req_qubits} qubits): ")
            results_data = run_incrementer_circuit_ibm_runtime(NUM_COUNTER_QUBITS, initial_states_to_test, use_simulator=False, backend_name=backend_input)
        else: print("Cannot run on IBM Quantum: QiskitRuntimeService not available.")
    else:
        print("Invalid option selected. Exiting.")

    # Final summary message
    print("-" * 60)
    if results_data:
        print("\n--- Experiment Summary (Binary Incrementer) ---")
        print("- Circuit Description:")
        print(f"  - Initializes {NUM_COUNTER_QUBITS} qubits to an initial state.")
        print(f"  - Applies X, CX, CCX... gates using ripple-carry logic to implement |x> -> |x+1 mod {2**NUM_COUNTER_QUBITS}>.")
        print(f"  - Measures all {NUM_COUNTER_QUBITS} qubits.")
        print("- Expected Ideal Outcome:")
        print("  - Measurement outcome corresponds to the binary representation of (initial_state + 1).")
        print("\nExecution finished.")
    else:
        print("\nExecution finished, but no results were generated (check for errors above).")
    print("-" * 60)

