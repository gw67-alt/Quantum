



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
    token='a4179ea065816da308a9d745ac63b5c472deddb3b2274d0bcdec33ce68d80b507a0a6b68ad2d1e0301440d8bf51956ef64831c006e3e1066dd73e586374a934a',
    channel='ibm_quantum',  # Specify the channel
    instance='ibm-q/open/main',  # This is the default open instance
    overwrite=True
)
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
try:
    from qiskit_ibm_runtime import (
        QiskitRuntimeService,
        Sampler,             # SamplerV2 primitive
        Session,
        SamplerOptions       # Specific options class for SamplerV2
    )
except ImportError as e:
      print(f"Error importing qiskit_ibm_runtime components: {e}")
      # Set placeholders
      QiskitRuntimeService = None
      Sampler = None
      Session = None
      SamplerOptions = None

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
    if input_AB[1] == '0': qc.x(in_B)
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
# IBM Runtime Execution Function (MODIFIED FOR AND CIRCUIT)
# =============================================================================
def run_and_circuit_ibm_runtime(use_simulator=True, backend_name=None):
    """
    MODIFIED: Runs the Phase-Based AND circuit using IBM Runtime.
    Calls create_and_circuit_phase_based and analyze_and_results.
    Assumes CCZ gate is allowed.
    """
    if not QiskitRuntimeService or not Sampler or not Session or not SamplerOptions: return {}
    print("\n--- Running Phase-Based AND circuit on IBM Quantum ---")
    print("*** Assumes CCZ gate is permitted ***")
    try:
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
    options = SamplerOptions(optimization_level=1, resilience_level=1)
    try: options.environment.log_level = "WARNING"
    except AttributeError: pass
    print(f"Using SamplerOptions: optimization_level={options.optimization_level}, resilience_level={options.resilience_level}")

    # --- Session and Sampler Execution ---
    try:
        print(f"Attempting session with backend: {actual_backend_name}")
        with Session(backend=backend_object) as session:
            print(f"Session active (ID: {session.session_id})")
            sampler = Sampler(session=session, options=options)
            print(f"SamplerV2 initialized.")
            num_shots = 8192

            circuits_to_run = []; circuit_inputs = []
            print("\nPreparing and transpiling circuits...")
            for input_AB in input_combinations:
                # *** CALL THE AND CIRCUIT FUNCTION ***
                qc = create_and_circuit_phase_based(input_AB)
                try:
                    # Transpilation handles CCZ decomposition if needed
                    transpiled_qc = transpile(qc, backend=backend_object, optimization_level=options.optimization_level)
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
        if results: plt.show()
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

    if results: plt.show()
    print("\n--- Local Simulation Finished (Phase-Based AND) ---")
    return results

# =============================================================================
# Main Execution Block (MODIFIED FOR AND CIRCUIT)
# =============================================================================
if __name__ == "__main__":

    # --- Optional IBM Quantum Account Setup ---

    print("Quantum Phase-Based AND Simulation")
    print("=" * 60)
    # Timestamp and location context
    now_local = datetime.datetime.now(datetime.timezone.utc).astimezone()
    print(f"Current Time: {now_local.strftime('%Y-%m-%d %H:%M:%S %Z%z')}")
    print(f"Location Context: Golden Square, Victoria, Australia")
    print("(Computes A AND B using H-CCZ-H phase encoding)")
    print("(Output=1 ('high') if A=1, B=1; Output=0 ('low') otherwise)")
    print("-" * 60)

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
        # *** CALL MODIFIED LOCAL SIM ***
        results_data = simulate_and_circuit_locally()
    elif run_option == '2':
        if QiskitRuntimeService:
             # *** CALL MODIFIED RUNTIME SIM ***
             results_data = run_and_circuit_ibm_runtime(use_simulator=True)
        else: print("Cannot run on IBM Quantum: QiskitRuntimeService not available.")
    elif run_option == '3':
         if QiskitRuntimeService:
             # *** CALL MODIFIED RUNTIME HW (AUTO) ***
             results_data = run_and_circuit_ibm_runtime(use_simulator=False, backend_name=None)
         else: print("Cannot run on IBM Quantum: QiskitRuntimeService not available.")
    elif run_option == '4':
        if QiskitRuntimeService:
            backend_input = input("Enter the specific IBM Quantum backend name (e.g., ibm_brisbane): ")
            # *** CALL MODIFIED RUNTIME HW (MANUAL) ***
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
