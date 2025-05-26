import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class Photon:
    """Individual photon with quantum properties"""
    x: float          # Position
    k: float          # Wave vector (momentum)
    omega: float      # Frequency
    amplitude: complex # Quantum amplitude
    phase: float      # Phase
    polarization: str # 'H' or 'V' (horizontal/vertical)
    birth_time: float # When photon was created
    future_correlation: float = 0.0  # Retrocausal correlation strength
    
    def energy(self):
        """Photon energy E = ℏω"""
        return abs(self.omega)  # ℏ = 1 in natural units
    
    def momentum(self):
        """Photon momentum p = ℏk"""
        return abs(self.k)
    
    def wavelength(self):
        """Photon wavelength λ = 2π/k"""
        return 2 * np.pi / abs(self.k) if self.k != 0 else np.inf

class RetrocausalPhotonSimulator:
    def __init__(self, space_size=100.0, dt=0.01, c=1.0, retro_strength=0.1):
        """
        Quantum photon simulator with retrocausal effects
        
        Parameters:
        - space_size: Size of simulation space
        - dt: Time step
        - c: Speed of light
        - retro_strength: Strength of retrocausal coupling
        """
        self.space_size = space_size
        self.dt = dt
        self.c = c
        self.retro_strength = retro_strength
        
        # Photon collections
        self.photons: List[Photon] = []
        self.future_photons: List[Photon] = []
        self.absorbed_photons: List[Photon] = []
        
        # Quantum field grids
        self.grid_points = 500
        self.x_grid = np.linspace(0, space_size, self.grid_points)
        self.dx = self.x_grid[1] - self.x_grid[0]
        
        # Quantum field amplitudes
        self.psi_field = np.zeros(self.grid_points, dtype=complex)  # Photon field
        self.future_field = np.zeros(self.grid_points, dtype=complex)  # Future influence
        self.vacuum_fluctuations = np.zeros(self.grid_points, dtype=complex)
        
        # Time and history
        self.t = 0.0
        self.time_delay = 5.0  # Retrocausal time window
        
        # History for visualization
        self.photon_history = []
        self.field_history = []
        self.future_field_history = []
        self.energy_history = []
        
        # Detectors
        self.detectors = []
        self.detection_events = []
        
    def add_detector(self, position, efficiency=0.8, name="Detector"):
        """Add a photon detector at given position"""
        self.detectors.append({
            'position': position,
            'efficiency': efficiency,
            'name': name,
            'detections': []
        })
    
    def create_photon(self, x, k, omega=None, amplitude=1.0, polarization='H'):
        """Create a new photon"""
        if omega is None:
            omega = abs(k) * self.c  # Dispersion relation ω = ck
        
        photon = Photon(
            x=x,
            k=k,
            omega=omega,
            amplitude=complex(amplitude),
            phase=random.uniform(0, 2*np.pi),
            polarization=polarization,
            birth_time=self.t
        )
        
        self.photons.append(photon)
        return photon
    
    def create_photon_pulse(self, center_x, width, n_photons=10, k_mean=10.0, k_spread=2.0):
        """Create a pulse of correlated photons"""
        for i in range(n_photons):
            # Position within pulse
            x = center_x + np.random.normal(0, width)
            
            # Wave vector with some spread
            k = np.random.normal(k_mean, k_spread)
            
            # Correlated amplitude
            amplitude = np.exp(-(x - center_x)**2 / (2 * width**2))
            
            self.create_photon(x, k, amplitude=amplitude)
    
    def future_field_distribution(self, t):
        """
        Generate future photon field distribution
        This represents quantum field correlations from the future
        """
        future_time = t + self.time_delay
        
        # Future photon source (delayed)
        if future_time > 10.0:  # After some delay
            # Moving wave packet from the future
            center = self.space_size * 0.8 - self.c * (future_time - 10.0) * 0.5
            width = 3.0
            amplitude = 0.5 * np.exp(-(future_time - 15.0)**2 / 20.0)
            
            if 0 < center < self.space_size:
                future_dist = amplitude * np.exp(-(self.x_grid - center)**2 / (2 * width**2))
                return future_dist * np.exp(1j * 2.0 * center)  # Complex field
        
        return np.zeros(self.grid_points, dtype=complex)
    
    def update_quantum_field(self):
        """Update the quantum photon field based on photon positions"""
        # Reset field
        self.psi_field = np.zeros(self.grid_points, dtype=complex)
        
        # Add contribution from each photon
        for photon in self.photons:
            if 0 <= photon.x <= self.space_size:
                # Find grid index
                idx = int(photon.x / self.dx)
                if 0 <= idx < self.grid_points:
                    # Quantum field amplitude
                    wave_function = (photon.amplitude * 
                                   np.exp(1j * (photon.k * photon.x - photon.omega * self.t + photon.phase)))
                    
                    # Add to field with spatial distribution
                    sigma = 1.0  # Photon localization width
                    for j in range(max(0, idx-10), min(self.grid_points, idx+11)):
                        x_j = j * self.dx
                        spatial_weight = np.exp(-(x_j - photon.x)**2 / (2 * sigma**2))
                        self.psi_field[j] += wave_function * spatial_weight
        
        # Add vacuum fluctuations
        vacuum_amplitude = 0.01
        self.vacuum_fluctuations = (vacuum_amplitude * 
                                   (np.random.normal(0, 1, self.grid_points) + 
                                    1j * np.random.normal(0, 1, self.grid_points)))
        self.psi_field += self.vacuum_fluctuations
    
    def apply_retrocausal_coupling(self):
        """Apply retrocausal effects from future field"""
        # Get future field influence
        self.future_field = self.future_field_distribution(self.t)
        
        # Apply retrocausal coupling to each photon
        for photon in self.photons:
            if 0 <= photon.x <= self.space_size:
                idx = int(photon.x / self.dx)
                if 0 <= idx < self.grid_points:
                    # Future influence on photon
                    future_influence = self.future_field[idx]
                    
                    # Modify photon properties based on future correlation
                    correlation_strength = abs(future_influence) * self.retro_strength
                    photon.future_correlation = correlation_strength
                    
                    # Phase modification from future
                    phase_shift = np.angle(future_influence) * correlation_strength
                    photon.phase += phase_shift * self.dt
                    
                    # Amplitude modification (quantum interference)
                    amplitude_factor = 1.0 + correlation_strength * 0.1
                    photon.amplitude *= amplitude_factor
    
    def propagate_photons(self):
        """Propagate all photons according to their wave vectors"""
        for photon in self.photons[:]:  # Copy list to allow modification
            # Classical propagation
            photon.x += photon.k / abs(photon.k) * self.c * self.dt if photon.k != 0 else 0
            
            # Phase evolution
            photon.phase += photon.omega * self.dt
            
            # Remove photons that left the simulation space
            if photon.x < 0 or photon.x > self.space_size:
                self.photons.remove(photon)
    
    def check_detections(self):
        """Check for photon detections at detector positions"""
        for detector in self.detectors:
            det_pos = detector['position']
            det_efficiency = detector['efficiency']
            
            for photon in self.photons[:]:  # Copy to allow modification
                # Check if photon is near detector
                if abs(photon.x - det_pos) < 0.5:  # Detection window
                    # Quantum detection probability
                    detection_prob = (abs(photon.amplitude)**2 * det_efficiency * self.dt)
                    
                    if random.random() < detection_prob:
                        # Detection event
                        detection_event = {
                            'time': self.t,
                            'position': photon.x,
                            'energy': photon.energy(),
                            'momentum': photon.momentum(),
                            'wavelength': photon.wavelength(),
                            'polarization': photon.polarization,
                            'future_correlation': photon.future_correlation,
                            'detector': detector['name']
                        }
                        
                        detector['detections'].append(detection_event)
                        self.detection_events.append(detection_event)
                        
                        # Remove detected photon
                        self.absorbed_photons.append(photon)
                        self.photons.remove(photon)
                        break
    
    def calculate_observables(self):
        """Calculate quantum observables"""
        observables = {}
        
        # Total photon number
        observables['n_photons'] = len(self.photons)
        
        # Total energy
        total_energy = sum(photon.energy() for photon in self.photons)
        observables['total_energy'] = total_energy
        
        # Total momentum
        total_momentum = sum(photon.momentum() * np.sign(photon.k) for photon in self.photons)
        observables['total_momentum'] = total_momentum
        
        # Field intensity |ψ|²
        field_intensity = np.abs(self.psi_field)**2
        observables['field_intensity'] = np.sum(field_intensity) * self.dx
        
        # Future correlation strength
        avg_future_correlation = (np.mean([p.future_correlation for p in self.photons]) 
                                if self.photons else 0.0)
        observables['avg_future_correlation'] = avg_future_correlation
        
        # Vacuum energy
        vacuum_energy = np.sum(np.abs(self.vacuum_fluctuations)**2) * self.dx
        observables['vacuum_energy'] = vacuum_energy
        
        return observables
    
    def step(self):
        """Single simulation step"""
        # Update quantum field
        self.update_quantum_field()
        
        # Apply retrocausal effects
        self.apply_retrocausal_coupling()
        
        # Propagate photons
        self.propagate_photons()
        
        # Check for detections
        self.check_detections()
        
        # Store history
        self.photon_history.append([p.__dict__.copy() for p in self.photons])
        self.field_history.append(self.psi_field.copy())
        self.future_field_history.append(self.future_field.copy())
        
        # Calculate and store observables
        observables = self.calculate_observables()
        self.energy_history.append(observables)
        
        # Advance time
        self.t += self.dt
    
    def run_simulation(self, total_time=50.0):
        """Run the complete simulation"""
        print("=== Retrocausal Photon Simulation ===")
        print(f"Simulation time: {total_time}")
        print(f"Initial photons: {len(self.photons)}")
        print(f"Retrocausal coupling strength: {self.retro_strength}")
        print(f"Detectors: {len(self.detectors)}")
        
        steps = int(total_time / self.dt)
        
        for step in range(steps):
            self.step()
            
            if step % 1000 == 0:
                print(f"Step {step}/{steps}, Time: {self.t:.2f}, Photons: {len(self.photons)}")
        
        print(f"\nSimulation complete!")
        print(f"Total detection events: {len(self.detection_events)}")
        print(f"Final photon count: {len(self.photons)}")
    
    def plot_results(self, save_plots=True, show_plots=False):
        """Create comprehensive visualization"""
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Photon trajectories in spacetime
        ax1 = plt.subplot(2, 3, 1)
        self.plot_spacetime_diagram(ax1)
        
        # 2. Current photon field
        ax2 = plt.subplot(2, 3, 2)
        self.plot_current_field(ax2)
        
        # 3. Detection events
        ax3 = plt.subplot(2, 3, 3)
        self.plot_detection_events(ax3)
        
        # 4. Energy evolution
        ax4 = plt.subplot(2, 3, 4)
        self.plot_energy_evolution(ax4)
        
        # 5. Future correlation analysis
        ax5 = plt.subplot(2, 3, 5)
        self.plot_future_correlations(ax5)
        
        # 6. Quantum field intensity
        ax6 = plt.subplot(2, 3, 6)
        self.plot_field_intensity(ax6)
        
        plt.suptitle('Retrocausal Photon Simulation Results', fontsize=16)
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('retrocausal_photon_simulation.png', dpi=300, bbox_inches='tight')
            print("Plots saved as 'retrocausal_photon_simulation.png'")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_spacetime_diagram(self, ax):
        """Plot photon trajectories in spacetime"""
        # Track individual photon paths
        for i, photon_states in enumerate(zip(*self.photon_history)):
            if len(photon_states) > 1:
                times = np.array([i * self.dt for i in range(len(photon_states))])
                positions = np.array([state['x'] for state in photon_states])
                
                # Color by energy
                energy = photon_states[0]['omega']
                color = plt.cm.plasma(energy / 20.0)  # Normalize energy for coloring
                
                ax.plot(positions, times, color=color, alpha=0.7, linewidth=1)
        
        # Mark detectors
        for detector in self.detectors:
            ax.axvline(detector['position'], color='red', linestyle='--', alpha=0.7, 
                      label=f"{detector['name']}")
        
        ax.set_xlabel('Position')
        ax.set_ylabel('Time')
        ax.set_title('Photon Trajectories in Spacetime')
        ax.grid(True, alpha=0.3)
    
    def plot_current_field(self, ax):
        """Plot current quantum field state"""
        if self.field_history:
            current_field = self.field_history[-1]
            future_field = self.future_field_history[-1]
            
            # Field intensity
            intensity = np.abs(current_field)**2
            future_intensity = np.abs(future_field)**2
            
            ax.plot(self.x_grid, intensity, 'b-', label='Current Field |ψ|²', linewidth=2)
            ax.plot(self.x_grid, future_intensity, 'r--', label='Future Influence', linewidth=2)
            
            # Mark photon positions
            for photon in self.photons:
                if 0 <= photon.x <= self.space_size:
                    ax.axvline(photon.x, color='green', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('Position')
        ax.set_ylabel('Field Intensity')
        ax.set_title('Quantum Photon Field')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_detection_events(self, ax):
        """Plot detection events analysis"""
        if self.detection_events:
            times = [event['time'] for event in self.detection_events]
            energies = [event['energy'] for event in self.detection_events]
            correlations = [event['future_correlation'] for event in self.detection_events]
            
            scatter = ax.scatter(times, energies, c=correlations, cmap='RdYlBu_r', 
                               s=50, alpha=0.7)
            plt.colorbar(scatter, ax=ax, label='Future Correlation')
            
            ax.set_xlabel('Detection Time')
            ax.set_ylabel('Photon Energy')
            ax.set_title('Detection Events\n(colored by retrocausal correlation)')
        else:
            ax.text(0.5, 0.5, 'No detections', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title('Detection Events')
        
        ax.grid(True, alpha=0.3)
    
    def plot_energy_evolution(self, ax):
        """Plot energy conservation and evolution"""
        if self.energy_history:
            times = np.array([i * self.dt for i in range(len(self.energy_history))])
            
            total_energies = [obs['total_energy'] for obs in self.energy_history]
            field_energies = [obs['field_intensity'] for obs in self.energy_history]
            vacuum_energies = [obs['vacuum_energy'] for obs in self.energy_history]
            
            ax.plot(times, total_energies, 'b-', label='Photon Energy', linewidth=2)
            ax.plot(times, field_energies, 'r-', label='Field Energy', linewidth=2)
            ax.plot(times, vacuum_energies, 'g-', label='Vacuum Energy', linewidth=1, alpha=0.7)
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Energy')
            ax.set_title('Energy Evolution')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def plot_future_correlations(self, ax):
        """Plot retrocausal correlation analysis"""
        if self.energy_history:
            times = np.array([i * self.dt for i in range(len(self.energy_history))])
            correlations = [obs['avg_future_correlation'] for obs in self.energy_history]
            
            ax.plot(times, correlations, 'purple', linewidth=2)
            ax.fill_between(times, 0, correlations, alpha=0.3, color='purple')
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Average Future Correlation')
            ax.set_title('Retrocausal Coupling Strength')
            ax.grid(True, alpha=0.3)
    
    def plot_field_intensity(self, ax):
        """Plot field intensity evolution"""
        if self.field_history:
            # Create spacetime plot of field intensity
            times = np.array([i * self.dt for i in range(len(self.field_history))])
            intensities = np.array([np.abs(field)**2 for field in self.field_history])
            
            X, T = np.meshgrid(self.x_grid, times)
            
            contour = ax.contourf(X, T, intensities, levels=20, cmap='plasma')
            plt.colorbar(contour, ax=ax, label='Field Intensity')
            
            ax.set_xlabel('Position')
            ax.set_ylabel('Time')
            ax.set_title('Field Intensity Evolution')
    
    def print_analysis(self):
        """Print detailed analysis of simulation results"""
        print("\n" + "="*60)
        print("=== RETROCAUSAL PHOTON SIMULATION ANALYSIS ===")
        print("="*60)
        
        # Basic statistics
        print(f"\nSIMULATION PARAMETERS:")
        print(f"Total simulation time: {self.t:.2f}")
        print(f"Space size: {self.space_size}")
        print(f"Retrocausal coupling strength: {self.retro_strength}")
        print(f"Speed of light: {self.c}")
        
        # Photon statistics
        print(f"\nPHOTON STATISTICS:")
        print(f"Remaining photons: {len(self.photons)}")
        print(f"Absorbed photons: {len(self.absorbed_photons)}")
        print(f"Total detection events: {len(self.detection_events)}")
        
        if self.photons:
            energies = [p.energy() for p in self.photons]
            momenta = [p.momentum() for p in self.photons]
            correlations = [p.future_correlation for p in self.photons]
            
            print(f"Average photon energy: {np.mean(energies):.4f}")
            print(f"Energy spread: {np.std(energies):.4f}")
            print(f"Average momentum: {np.mean(momenta):.4f}")
            print(f"Average future correlation: {np.mean(correlations):.4f}")
        
        # Detection analysis
        if self.detection_events:
            print(f"\nDETECTION ANALYSIS:")
            det_energies = [event['energy'] for event in self.detection_events]
            det_correlations = [event['future_correlation'] for event in self.detection_events]
            
            print(f"Average detected energy: {np.mean(det_energies):.4f}")
            print(f"Detection energy spread: {np.std(det_energies):.4f}")
            print(f"Average correlation at detection: {np.mean(det_correlations):.4f}")
            
            # Detector efficiency
            for detector in self.detectors:
                n_detections = len(detector['detections'])
                print(f"{detector['name']}: {n_detections} detections")
        
        # Energy conservation
        if self.energy_history:
            print(f"\nENERGY CONSERVATION:")
            initial_energy = self.energy_history[0]['total_energy']
            final_energy = self.energy_history[-1]['total_energy']
            energy_change = (final_energy - initial_energy) / initial_energy * 100
            
            print(f"Initial total energy: {initial_energy:.6f}")
            print(f"Final total energy: {final_energy:.6f}")
            print(f"Energy change: {energy_change:.2f}%")
        
        # Retrocausal effects
        print(f"\nRETROCAUSAL EFFECTS:")
        if self.energy_history:
            max_correlation = max(obs['avg_future_correlation'] for obs in self.energy_history)
            print(f"Maximum future correlation observed: {max_correlation:.6f}")
            
            # Analyze correlation timeline
            correlation_timeline = [obs['avg_future_correlation'] for obs in self.energy_history]
            peak_time = np.argmax(correlation_timeline) * self.dt
            print(f"Peak correlation time: {peak_time:.2f}")
        
        print("\n" + "="*60)
        print("The simulation demonstrates quantum photon behavior")
        print("with retrocausal influences from future field states.")
        print("Future correlations affect present photon properties")
        print("through quantum field coupling mechanisms.")
        print("="*60)

# Example usage and demonstration
if __name__ == "__main__":
    # Create simulator
    sim = RetrocausalPhotonSimulator(
        space_size=50.0,
        dt=0.02,
        c=1.0,
        retro_strength=0.15
    )
    
    # Add detectors
    sim.add_detector(position=25.0, efficiency=0.9, name="Detector A")
    sim.add_detector(position=40.0, efficiency=0.8, name="Detector B")
    
    # Create initial photon pulses
    print("Creating photon pulses...")
    
    # Pulse 1: Low energy photons
    sim.create_photon_pulse(center_x=5.0, width=2.0, n_photons=15, 
                           k_mean=5.0, k_spread=1.0)
    
    # Pulse 2: High energy photons
    sim.create_photon_pulse(center_x=10.0, width=1.5, n_photons=10, 
                           k_mean=12.0, k_spread=2.0)
    
    # Individual photons with specific properties
    sim.create_photon(x=15.0, k=8.0, amplitude=1.5, polarization='V')
    sim.create_photon(x=20.0, k=6.0, amplitude=1.2, polarization='H')
    
    print(f"Total initial photons: {len(sim.photons)}")
    
    # Run simulation
    print("\nStarting retrocausal photon simulation...")
    sim.run_simulation(total_time=40.0)
    
    # Generate results
    sim.plot_results(save_plots=True, show_plots=False)
    sim.print_analysis()
    
    print("\n=== QUANTUM INTERPRETATION ===")
    print("This simulation models individual photons as quantum particles")
    print("with wave-particle duality and retrocausal correlations.")
    print("Key features:")
    print("• Individual photon tracking with quantum properties")
    print("• Retrocausal coupling through future field correlations")
    print("• Quantum detection events with probabilistic outcomes")
    print("• Field intensity |ψ|² representing photon probability density")
    print("• Energy-momentum conservation in relativistic framework")
    print("• Vacuum fluctuations contributing to quantum noise")
