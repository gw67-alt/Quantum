# =============================================================================
# Imports
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np
import traceback # For detailed error printing
import datetime # For timestamp

# Qiskit Core
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import CXGate, CCXGate # For checking decomposition later if needed
from qiskit.visualization import plot_histogram # Ensure plot_histogram is imported

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
        Sampler,             # SamplerV2 primitive
        Session,
        SamplerOptions       # Specific options class for SamplerV2 (though may not be used directly below)
    )
except ImportError as e:
      print(f"Error importing qiskit_ibm_runtime components: {e}")
      # Set placeholders
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
# Circuit Creation Function (Phase-Based AND)
# =============================================================================
def create_and_circuit_phase_based(input_AB="00"):
    """
    Creates a 3-qubit circuit that computes the logical AND of q[0] and q[1]
    into q[2] using phase encoding (H-CCZ-H).

    Output q[2] should be 1 ('high') iff input is '11', otherwise 0 ('low').
    Assumes CCZ gate is permitted.

    Qubits:
    q[0]: Input 'A' (Control 1 for CCZ)
    q[1]: Input 'B' (Control 2 for CCZ)
    q[2]: Readout 'C' (Target for CCZ)

    Parameters:
    input_AB (str): 2-bit string "xy" defining initial state |x>|y> for q[0], q[1].

    Returns:
    QuantumCircuit: The constructed circuit (3 qubits, 1 classical bit).
    """
    if len(input_AB) != 2 or not all(bit in '01' for bit in input_AB):
        raise ValueError("input_AB must be a 2-bit string (e.g., '01')")

    qreg = QuantumRegister(3, 'q')
    creg = ClassicalRegister(1, 'c') # Measure only q[2]
    qc = QuantumCircuit(qreg, creg, name=f"PhaseAND_In_{input_AB}")

    in_A = qreg[0] # Input A
    in_B = qreg[1] # Input B
    readout_C = qreg[2] # Readout C

    # Step 1: Initialize Input States
    if input_AB[0] == '1': qc.x(in_A)
    if input_AB[1] == '1': qc.x(in_B)
    # readout_C starts as |0>
    qc.barrier(label="Inputs")

    # Step 2: Prepare Readout Qubit and Apply CCZ
    qc.h(readout_C)                     # Put readout qubit into |+> state
    qc.ccz(in_A, in_B, readout_C)      # Apply CCZ gate: phase flips if in_A=1 and in_B=1
    qc.barrier(label="H-CCZ")

    # Step 3: Final Hadamard on Readout Qubit
    qc.h(readout_C)                     # Apply H again to convert phase back to Z-basis state
                                       # |+> -(No CCZ)-> |+> -(H)-> |0>  (AND=0)
                                       # |+> -(CCZ)   -> |-> -(H)-> |1>  (AND=1)
    qc.barrier(label="Final H")

    # Step 4: Measure the Readout Qubit 'C'
    qc.measure(readout_C, creg[0]) # Measure q[2] -> c[0]

    return qc

# =============================================================================
# Results Analysis Function (for AND Circuit)
# =============================================================================
def analyze_and_results(counts, input_AB):
    """
    Analyzes measurement results for the 1-bit output of the AND circuit.
    Checks if the output matches the expected AND result.

    Parameters:
    counts (dict): Measurement counts {'0': count0, '1': count1}.
    input_AB (str): The 2-bit input state string "xy".
    """
    in_A = int(input_AB[0])
    in_B = int(input_AB[1])
    print(f"\nResults analysis for input state |{in_A}>|{in_B}>:")

    if not counts: print("No counts received."); return

    print("\nReadout qubit 'C' (q[2]) outcomes:")
    print("Outcome | Count | Probability | Expected (A AND B)")
    print("-" * 50) # Adjusted width

    # Normalize keys and sum counts
    normalized_counts = {'0': 0, '1': 0}; total_shots = 0
    for key, count in counts.items():
        total_shots += count
        try:
            outcome_int = int(str(key), 0)
            if outcome_int == 0: normalized_counts['0'] += count
            elif outcome_int == 1: normalized_counts['1'] += count
        except (ValueError, TypeError): print(f"Warning: Could not interpret result key '{key}'."); continue

    if total_shots == 0: print("Total shots zero."); return

    # Expected AND result: 1 iff A=1 AND B=1
    expected_outcome_val = 1 if (in_A == 1 and in_B == 1) else 0
    expected_outcome_str = str(expected_outcome_val)

    for outcome in ['0', '1']:
        count = normalized_counts[outcome]
        prob = count / total_shots if total_shots > 0 else 0
        is_expected = "<- Expected" if outcome == expected_outcome_str else ""
        print(f"{outcome:^7s} | {count:5d} | {prob:.4f}           | {is_expected}")

    # Interpretation
    if total_shots > 0:
      majority_outcome = max(normalized_counts, key=normalized_counts.get)
      if normalized_counts[majority_outcome] / total_shots > 0.6: # Confidence threshold
          if majority_outcome == expected_outcome_str: print(f"Interpretation: Readout qubit correctly calculated AND = {expected_outcome_val}.")
          else: print(f"Interpretation: Readout qubit INCORRECTLY measured {majority_outcome} (expected {expected_outcome_str}). Noise likely.")
      elif abs(normalized_counts.get('0', 0)/total_shots - 0.5) < 0.15 :
           print(f"Interpretation: Results are mixed (~50/50). Expected {expected_outcome_str}. Potential issue.")
      else: print(f"Interpretation: No clear majority outcome. Expected {expected_outcome_str}.")
    else: print("Interpretation: No valid counts.")
    print(f"(Note: Circuit implements '{in_A} AND {in_B}')")


# =============================================================================
# IBM Runtime Execution Function (Options Fix 5 Applied - Sampler Init)
# =============================================================================
def run_and_circuit_ibm_runtime(use_simulator=True, backend_name=None):
    """
    MODIFIED: Runs the Phase-Based AND circuit using IBM Runtime.
    Calls create_and_circuit_phase_based and analyze_and_results.
    Assumes CCZ gate is allowed. Relies on default Sampler options
    and implicit Session context.
    """
    if not QiskitRuntimeService or not Sampler or not Session: return {} # Removed SamplerOptions check
    print("\n--- Running Phase-Based AND circuit on IBM Quantum ---")
    print("*** Assumes CCZ gate is permitted ***")
    try:
        # Assumes account is saved via QiskitRuntimeService.save_account()
        service = QiskitRuntimeService()
        print(f"Initialized QiskitRuntimeService (Account: {service.active_account()})")
    except Exception as e: print(f"Failed init: {e}"); return {}

    # --- Backend Selection ---
    actual_backend_name = None; backend_object = None; min_qubits_needed = 3 # Circuit needs 3 qubits
    print("Selecting backend...")
    try:
        if use_simulator:
            sim_backend_name = "ibmq_qasm_simulator"; backend_object = service.backend(sim_backend_name)
        else: # Hardware
            if backend_name is None: # Find least busy
                backends = service.backends(operational=True, simulator=False, min_num_qubits=min_qubits_needed)
                if not backends: raise RuntimeError(f"No suitable hardware ({min_qubits_needed}+ qubits) found.")
                def get_pending_jobs(b):
                    try: return b.status().pending_jobs
                    except: return float('inf')
                backend_object = min(backends, key=get_pending_jobs)
            else: # Use specified name
                backend_object = service.backend(backend_name)
                if backend_object.num_qubits < min_qubits_needed: raise RuntimeError(f"Backend '{backend_name}' needs >= {min_qubits_needed} qubits.")
                if not backend_object.operational: raise RuntimeError(f"Backend '{backend_name}' not operational.")
        actual_backend_name = backend_object.name
        print(f"Using backend: {actual_backend_name}")
    except Exception as e: print(f"Backend selection error: {e}"); traceback.print_exc(); return {}
    if backend_object is None: print("Failed getting backend."); return {}
    # --- End Backend Selection ---

    # Test all 4 input states for q[0]q[1]
    input_combinations = [format(i, '02b') for i in range(4)]
    results = {}
    print(f"\nRunning circuits on backend: {actual_backend_name}")

    # --- Options Setup ---
    # REMOVED explicit SamplerOptions creation due to version compatibility issues.
    # Relying on default Sampler resilience settings (usually level 1).
    # Optimization level for transpile is set directly in the transpile() call.
    print("Relying on default Sampler resilience settings.")
    # --- End Options Setup ---


    # --- Session and Sampler Execution ---
    try:
        print(f"Attempting session with backend: {actual_backend_name}")
        with Session(backend=backend_object) as session:
            print(f"Session active (ID: {session.session_id})")
            # *** Initialize Sampler without explicit session or options ***
            # It should automatically use the session from the 'with' block
            # and use default options (including resilience).
            sampler = Sampler()
            print(f"SamplerV2 initialized (using default options and active session).")
            num_shots = 8192

            circuits_to_run = []; circuit_inputs = []
            print("\nPreparing and transpiling circuits...")
            # Define optimization level for transpilation here
            transpile_optimization_level = 1
            print(f"Using transpilation optimization_level={transpile_optimization_level}")

            for input_AB in input_combinations:
                # *** CALL THE AND CIRCUIT FUNCTION ***
                qc = create_and_circuit_phase_based(input_AB)
                try:
                    # Transpilation handles CCZ decomposition if needed
                    # *** Pass optimization_level directly to transpile ***
                    transpiled_qc = transpile(qc, backend=backend_object,
                                              optimization_level=transpile_optimization_level)
                    circuits_to_run.append(transpiled_qc)
                    circuit_inputs.append(input_AB) # Track input state
                    print(f"  - Circuit for input |{input_AB}> prepared and transpiled.")
                except Exception as transpile_error:
                    print(f"  !!! Error transpiling input |{input_AB}>: {transpile_error}")

            if not circuits_to_run: return {}

            print(f"\nSubmitting batch job ({len(circuits_to_run)} circuits, {num_shots} shots each)...")
            job = sampler.run(circuits_to_run, shots=num_shots)
            print(f"Job submitted (ID: {job.job_id()}). Waiting...")
            primitive_result = job.result()
            print("Job completed.")

            print("\nProcessing results...")
            for i, pub_result in enumerate(primitive_result):
                 if i >= len(circuit_inputs): continue
                 current_input = circuit_inputs[i]
                 if not hasattr(pub_result, 'data') or not hasattr(pub_result.data, 'c'): continue
                 data_bin = pub_result.data.c
                 run_counts = data_bin.get_counts()
                 print(f"\n--- Extracted counts for Input State |{current_input}> ---")
                 results[current_input] = run_counts
                 # *** CALL THE AND ANALYSIS FUNCTION ***
                 analyze_and_results(run_counts, current_input)

                 if run_counts:
                     plot_counts = {str(k): v for k, v in run_counts.items()}
                     plot_histogram(plot_counts, sort='value',
                                    title=f"PhaseAND Results Input |{current_input}> ({actual_backend_name})")
                     plt.tight_layout(); plt.show(block=False)

    # Error Handling
    except TypeError as te: print(f"\nAPI mismatch (TypeError): {te}"); traceback.print_exc(); return {}
    except ValueError as ve: print(f"\nValue error: {ve}"); traceback.print_exc(); return {}
    except Exception as e: print(f"\nUnexpected error: {e}"); traceback.print_exc(); return {}
    finally:
        if results:
             print("\nFinalizing plots...")
             plt.show()
        print("--- Session block finished ---")

    print("\n--- IBM Quantum Run Finished (Phase-Based AND) ---")
    return results

# =============================================================================
# Local Simulation Function (MODIFIED FOR AND CIRCUIT)
# =============================================================================
def simulate_and_circuit_locally():
    """
    MODIFIED: Runs the Phase-Based AND circuit using qiskit-aer.
    Calls create_and_circuit_phase_based and analyze_and_results.
    """
    print("\n--- Running Phase-Based AND circuit Locally using qiskit-aer ---")
    print("*** Assumes CCZ gate is permitted ***")
    if Aer is None: print("qiskit-aer not found."); return {}
    try: simulator = Aer.get_backend('qasm_simulator')
    except Exception as e: print(f"Could not get qasm_simulator: {e}"); return {}

    input_combinations = [format(i, '02b') for i in range(4)]
    results = {}
    print("Preparing and simulating circuits locally...")
    for input_AB in input_combinations:
        # *** CALL THE AND CIRCUIT FUNCTION ***
        qc = create_and_circuit_phase_based(input_AB)
        shots = 8192
        print(f"  - Simulating circuit for input |{input_AB}> with {shots} shots...")
        job = simulator.run(qc, shots=shots)
        result = job.result()
        counts = result.get_counts(qc)
        print("  - Simulation complete.")
        results[input_AB] = counts
        # *** CALL THE AND ANALYSIS FUNCTION ***
        analyze_and_results(counts, input_AB)
        if counts:
            plot_counts = {str(k): v for k, v in counts.items()}
            plot_histogram(plot_counts, sort='value',
                           title=f"Local Sim PhaseAND Results Input |{input_AB}>")
            plt.tight_layout(); plt.show(block=False)

    if results:
        print("\nFinalizing plots...")
        plt.show()
    print("\n--- Local Simulation Finished (Phase-Based AND) ---")
    return results

# =============================================================================
# Main Execution Block (MODIFIED FOR AND CIRCUIT)
# =============================================================================
if __name__ == "__main__":

    # --- Optional IBM Quantum Account Setup ---
    # Ensure account is saved once beforehand if using IBM Quantum services.

    print("Quantum Phase-Based AND Simulation")
    print("=" * 60)
    # Timestamp and location context
    now_local = datetime.datetime.now(datetime.timezone.utc).astimezone()
    print(f"Current Time: {now_local.strftime('%Y-%m-%d %H:%M:%S %Z%z')}")
    print("(Computes A AND B using H-CCZ-H phase encoding)")
    print("(Output=1 ('high') if A=1, B=1; Output=0 ('low') otherwise)")
    print("-" * 60)
    # from your_script import create_and_circuit_phase_based

    # 2. Create an instance of the circuit (e.g., for input '11')
    input_example = "11"
    qc = create_and_circuit_phase_based(input_AB=input_example)

    # 3. Use the draw() method and print the result
    #    output='text' gives a text-based diagram
    #    fold controls line wrapping (adjust as needed)
    try:
        print(f"Circuit diagram for input |{input_example}>:")
        print(qc.draw(output='text', fold=80))
    except ImportError:
        print("Circuit drawing requires pylatexenc. Using basic print(qc):")
        print(qc)
    except Exception as e:
         print(f"An error occurred during drawing: {e}")
         print(qc) # Fallback
    run_option = input(
        "Choose run option:\n"
        "  1. Local Simulator (qiskit-aer, faster, ideal)\n"
        "  2. IBM Quantum Simulator (cloud, includes transpilation noise model)\n"
        "  3. IBM Quantum Hardware (Least Busy >= 3 qubits, real device)\n"
        "  4. IBM Quantum Hardware (Specify Name >= 3 qubits, real device)\n"
        "Enter option number (1-4): "
    )

    results_data = {} # Will store the counts dictionary

    if run_option == '1':
        results_data = simulate_and_circuit_locally()
    elif run_option == '2':
        if QiskitRuntimeService:
             results_data = run_and_circuit_ibm_runtime(use_simulator=True)
        else: print("Cannot run on IBM Quantum: QiskitRuntimeService not available.")
    elif run_option == '3':
         if QiskitRuntimeService:
             results_data = run_and_circuit_ibm_runtime(use_simulator=False, backend_name=None)
         else: print("Cannot run on IBM Quantum: QiskitRuntimeService not available.")
    elif run_option == '4':
        if QiskitRuntimeService:
            backend_input = input("Enter the specific IBM Quantum backend name (e.g., ibm_brisbane): ")
            results_data = run_and_circuit_ibm_runtime(use_simulator=False, backend_name=backend_input)
        else: print("Cannot run on IBM Quantum: QiskitRuntimeService not available.")
    else:
        print("Invalid option selected. Exiting.")

    # Final summary message
    print("-" * 60)
    if results_data:
        print("\n--- Experiment Summary (Phase-Based AND) ---")
        print("- Circuit Description:")
        print("  - Initializes q[0] (A), q[1] (B) to input state.")
        print("  - Computes A AND B into q[2] (C) using H-CCZ-H phase encoding.")
        print("- Expected Ideal Outcome:")
        print("  - Measurement of q[2] is '1' if input was |11>.")
        print("  - Measurement of q[2] is '0' if input was |00>, |01>, or |10>.")
        print("\nExecution finished.")
    else:
        print("\nExecution finished, but no results were generated (check for errors above).")
    print("-" * 60)
