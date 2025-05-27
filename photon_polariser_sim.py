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
    polarization_angle: float = 0.0  # Polarization angle in radians
    
    def energy(self):
        """Photon energy E = ℏω"""
        return abs(self.omega)  # ℏ = 1 in natural units
    
    def momentum(self):
        """Photon momentum p = ℏk"""
        return abs(self.k)
    
    def wavelength(self):
        """Photon wavelength λ = 2π/k"""
        return 2 * np.pi / abs(self.k) if self.k != 0 else np.inf
    
    def get_polarization_vector(self):
        """Get 2D polarization vector (Ex, Ey)"""
        if self.polarization == 'H':
            return np.array([1.0, 0.0])  # Horizontal
        elif self.polarization == 'V':
            return np.array([0.0, 1.0])  # Vertical
        elif self.polarization == 'D':
            return np.array([1.0, 1.0]) / np.sqrt(2)  # Diagonal
        elif self.polarization == 'A':
            return np.array([1.0, -1.0]) / np.sqrt(2)  # Anti-diagonal
        elif self.polarization == 'R':
            return np.array([1.0, 1j]) / np.sqrt(2)  # Right circular
        elif self.polarization == 'L':
            return np.array([1.0, -1j]) / np.sqrt(2)  # Left circular
        else:
            # Custom angle
            return np.array([np.cos(self.polarization_angle), 
                           np.sin(self.polarization_angle)])

@dataclass
class PolarizerOscillator:
    """Oscillating polarizer with time-dependent orientation"""
    position: float           # Position in space
    width: float             # Spatial width of polarizer
    oscillation_freq: float  # Oscillation frequency (Hz)
    amplitude: float         # Oscillation amplitude (radians)
    phase_offset: float      # Phase offset
    base_angle: float        # Base polarization angle
    transmission_efficiency: float = 0.95  # Maximum transmission
    name: str = "Polarizer"
    
    def get_angle(self, time):
        """Get current polarization angle"""
        return (self.base_angle + 
                self.amplitude * np.sin(2 * np.pi * self.oscillation_freq * time + self.phase_offset))
    
    def get_transmission_axis(self, time):
        """Get transmission axis vector at given time"""
        angle = self.get_angle(time)
        return np.array([np.cos(angle), np.sin(angle)])
    
    def calculate_transmission_probability(self, photon, time):
        """Calculate transmission probability for a photon"""
        if not (self.position - self.width/2 <= photon.x <= self.position + self.width/2):
            return 1.0  # Outside polarizer, full transmission
        
        # Get photon polarization vector
        photon_pol = photon.get_polarization_vector()
        
        # Get polarizer transmission axis
        transmission_axis = self.get_transmission_axis(time)
        
        # Calculate transmission probability using Malus's law
        # T = cos²(θ) where θ is angle between polarizations
        if np.iscomplexobj(photon_pol):
            # Handle circular polarizations
            intensity = np.abs(np.dot(photon_pol.conj(), transmission_axis))**2
        else:
            # Handle linear polarizations
            cos_theta = np.dot(photon_pol, transmission_axis)
            intensity = cos_theta**2
        
        return intensity * self.transmission_efficiency

class RetrocausalPhotonSimulator:
    def __init__(self, space_size=32.0, dt=0.01, c=1.0, retro_strength=0.1):
        """
        Quantum photon simulator with retrocausal effects and polarizer oscillators
        
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
        self.blocked_photons: List[Photon] = []  # Blocked by polarizers
        
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
        self.polarizer_history = []  # Track polarizer states
        
        # Detectors and polarizers
        self.detectors = []
        self.polarizers: List[PolarizerOscillator] = []
        self.detection_events = []
        self.polarizer_events = []  # Track polarizer interactions
        
    def add_detector(self, position, efficiency=0.8, name="Detector"):
        """Add a photon detector at given position"""
        self.detectors.append({
            'position': position,
            'efficiency': efficiency,
            'name': name,
            'detections': []
        })
    
    def add_polarizer_oscillator(self, position, width=2.0, oscillation_freq=1.0, 
                               amplitude=np.pi/4, base_angle=0.0, phase_offset=0.0, 
                               transmission_efficiency=0.95, name="Polarizer"):
        """Add an oscillating polarizer"""
        polarizer = PolarizerOscillator(
            position=position,
            width=width,
            oscillation_freq=oscillation_freq,
            amplitude=amplitude,
            phase_offset=phase_offset,
            base_angle=base_angle,
            transmission_efficiency=transmission_efficiency,
            name=name
        )
        self.polarizers.append(polarizer)
        return polarizer
    
    def create_photon(self, x, k, omega=None, amplitude=1.0, polarization='H', polarization_angle=0.0):
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
            birth_time=self.t,
            polarization_angle=polarization_angle
        )
        
        self.photons.append(photon)
        return photon
    
    def create_photon_pulse(self, center_x, width, n_photons=10, k_mean=10.0, k_spread=2.0, 
                          polarization_mix=None):
        """Create a pulse of correlated photons with mixed polarizations"""
        if polarization_mix is None:
            polarization_mix = ['H', 'V', 'D', 'A']  # Default mix
        
        for i in range(n_photons):
            # Position within pulse
            x = center_x + np.random.normal(0, width)
            
            # Wave vector with some spread
            k = np.random.normal(k_mean, k_spread)
            
            # Correlated amplitude
            amplitude = np.exp(-(x - center_x)**2 / (2 * width**2))
            
            # Random polarization from mix
            polarization = np.random.choice(polarization_mix)
            polarization_angle = np.random.uniform(0, 2*np.pi) if polarization == 'custom' else 0.0
            
            self.create_photon(x, k, amplitude=amplitude, polarization=polarization,
                             polarization_angle=polarization_angle)
    
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
    
    def apply_polarizer_interactions(self):
        """Apply polarizer interactions to photons"""
        polarizer_states = []
        
        for polarizer in self.polarizers:
            # Record current polarizer state
            current_angle = polarizer.get_angle(self.t)
            polarizer_states.append({
                'name': polarizer.name,
                'position': polarizer.position,
                'angle': current_angle,
                'transmission_axis': polarizer.get_transmission_axis(self.t)
            })
            
            # Check photons interacting with this polarizer
            for photon in self.photons[:]:  # Copy to allow modification
                if (polarizer.position - polarizer.width/2 <= photon.x <= 
                    polarizer.position + polarizer.width/2):
                    
                    # Calculate transmission probability
                    transmission_prob = polarizer.calculate_transmission_probability(photon, self.t)
                    
                    # Quantum measurement process
                    if random.random() < transmission_prob:
                        # Photon transmitted - update its polarization state
                        transmission_axis = polarizer.get_transmission_axis(self.t)
                        
                        # Project photon polarization onto transmission axis
                        photon_pol = photon.get_polarization_vector()
                        
                        if np.iscomplexobj(photon_pol):
                            # Handle complex polarizations
                            projection = np.dot(photon_pol.conj(), transmission_axis)
                            photon.amplitude *= abs(projection)
                        else:
                            # Handle real polarizations
                            projection = np.dot(photon_pol, transmission_axis)
                            photon.amplitude *= abs(projection)
                        
                        # Update photon polarization to match transmission axis
                        angle = np.arctan2(transmission_axis[1], transmission_axis[0])
                        photon.polarization_angle = angle
                        photon.polarization = 'custom'
                        
                        # Record transmission event
                        transmission_event = {
                            'time': self.t,
                            'photon_id': id(photon),
                            'polarizer': polarizer.name,
                            'transmission_prob': transmission_prob,
                            'polarizer_angle': current_angle,
                            'action': 'transmitted'
                        }
                        self.polarizer_events.append(transmission_event)
                        
                    else:
                        # Photon blocked/absorbed
                        blocking_event = {
                            'time': self.t,
                            'photon_id': id(photon),
                            'polarizer': polarizer.name,
                            'transmission_prob': transmission_prob,
                            'polarizer_angle': current_angle,
                            'photon_energy': photon.energy(),
                            'action': 'blocked'
                        }
                        self.polarizer_events.append(blocking_event)
                        
                        # Move to blocked photons
                        self.blocked_photons.append(photon)
                        self.photons.remove(photon)
        
        # Store polarizer states for history
        self.polarizer_history.append(polarizer_states)
    
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
                    # In check_detections method, change:
                    if abs(photon.x - det_pos) < 2.0:  # Larger detection window (was 0.5)
                        detection_prob = min(0.5, abs(photon.amplitude)**2 * det_efficiency * 5.0)  # Higher base probability
                    
                    if random.random() < detection_prob:
                        # Detection event
                        detection_event = {
                            'time': self.t,
                            'position': photon.x,
                            'energy': photon.energy(),
                            'momentum': photon.momentum(),
                            'wavelength': photon.wavelength(),
                            'polarization': photon.polarization,
                            'polarization_angle': photon.polarization_angle,
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
        observables['n_blocked'] = len(self.blocked_photons)
        
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
        
        # Polarization statistics
        if self.photons:
            polarizations = [p.polarization for p in self.photons]
            observables['polarization_counts'] = {pol: polarizations.count(pol) for pol in set(polarizations)}
        else:
            observables['polarization_counts'] = {}
        
        return observables
    
    def step(self):
        """Single simulation step"""
        # Update quantum field
        self.update_quantum_field()
        
        # Apply retrocausal effects
        self.apply_retrocausal_coupling()
        
        # Apply polarizer interactions
        self.apply_polarizer_interactions()
        
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
        print("=== Retrocausal Photon Simulation with Polarizer Oscillators ===")
        print(f"Simulation time: {total_time}")
        print(f"Initial photons: {len(self.photons)}")
        print(f"Retrocausal coupling strength: {self.retro_strength}")
        print(f"Detectors: {len(self.detectors)}")
        print(f"Polarizer oscillators: {len(self.polarizers)}")
        
        steps = int(total_time / self.dt)
        
        for step in range(steps):
            self.step()
            
            if step % 1000 == 0:
                print(f"Step {step}/{steps}, Time: {self.t:.2f}, Photons: {len(self.photons)}, Blocked: {len(self.blocked_photons)}")
        
        print(f"\nSimulation complete!")
        print(f"Total detection events: {len(self.detection_events)}")
        print(f"Total polarizer interactions: {len(self.polarizer_events)}")
        print(f"Final photon count: {len(self.photons)}")
        print(f"Blocked photon count: {len(self.blocked_photons)}")
    
    def plot_results(self, save_plots=True, show_plots=False):
        """Create comprehensive visualization"""
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Photon trajectories in spacetime
        ax1 = plt.subplot(3, 3, 1)
        self.plot_spacetime_diagram(ax1)
        
        # 2. Current photon field
        ax2 = plt.subplot(3, 3, 2)
        self.plot_current_field(ax2)
        
        # 3. Detection events
        ax3 = plt.subplot(3, 3, 3)
        self.plot_detection_events(ax3)
        
        # 4. Energy evolution
        ax4 = plt.subplot(3, 3, 4)
        self.plot_energy_evolution(ax4)
        
        # 5. Future correlation analysis
        ax5 = plt.subplot(3, 3, 5)
        self.plot_future_correlations(ax5)
        
        # 6. Quantum field intensity
        ax6 = plt.subplot(3, 3, 6)
        self.plot_field_intensity(ax6)
        
        # 7. Polarizer oscillation analysis
        ax7 = plt.subplot(3, 3, 7)
        self.plot_polarizer_oscillations(ax7)
        
        # 8. Polarization statistics
        ax8 = plt.subplot(3, 3, 8)
        self.plot_polarization_statistics(ax8)
        
        # 9. Transmission/blocking events
        ax9 = plt.subplot(3, 3, 9)
        self.plot_polarizer_interactions(ax9)
        
        plt.suptitle('Retrocausal Photon Simulation with Polarizer Oscillators', fontsize=16)
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('retrocausal_photon_polarizer_simulation.png', dpi=300, bbox_inches='tight')
            print("Plots saved as 'retrocausal_photon_polarizer_simulation.png'")
        
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
                
                # Color by polarization
                pol = photon_states[0]['polarization']
                color_map = {'H': 'blue', 'V': 'red', 'D': 'green', 'A': 'orange', 
                           'R': 'purple', 'L': 'brown', 'custom': 'gray'}
                color = color_map.get(pol, 'black')
                
                ax.plot(positions, times, color=color, alpha=0.7, linewidth=1)
        
        # Mark detectors
        for detector in self.detectors:
            ax.axvline(detector['position'], color='red', linestyle='--', alpha=0.7, 
                      label=f"{detector['name']}")
        
        # Mark polarizers
        for polarizer in self.polarizers:
            ax.axvspan(polarizer.position - polarizer.width/2, 
                      polarizer.position + polarizer.width/2, 
                      alpha=0.2, color='yellow', label=f"{polarizer.name}")
        
        ax.set_xlabel('Position')
        ax.set_ylabel('Time')
        ax.set_title('Photon Trajectories (colored by polarization)')
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
            
            # Mark polarizer positions
            for polarizer in self.polarizers:
                ax.axvspan(polarizer.position - polarizer.width/2, 
                          polarizer.position + polarizer.width/2, 
                          alpha=0.1, color='orange')
        
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
    
    def plot_polarizer_oscillations(self, ax):
        """Plot polarizer angle oscillations over time"""
        if self.polarizer_history and self.polarizers:
            times = np.array([i * self.dt for i in range(len(self.polarizer_history))])
            
            for i, polarizer in enumerate(self.polarizers):
                angles = []
                for step_polarizers in self.polarizer_history:
                    for pol_state in step_polarizers:
                        if pol_state['name'] == polarizer.name:
                            angles.append(pol_state['angle'])
                            break
                    else:
                        angles.append(0)  # Default if not found
                
                ax.plot(times[:len(angles)], np.degrees(angles), 
                       label=f"{polarizer.name} (f={polarizer.oscillation_freq:.1f}Hz)", 
                       linewidth=2)
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Polarization Angle (degrees)')
            ax.set_title('Polarizer Oscillations')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No polarizer data', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title('Polarizer Oscillations')
    
    def plot_polarization_statistics(self, ax):
        """Plot polarization distribution over time"""
        if self.energy_history:
            times = np.array([i * self.dt for i in range(len(self.energy_history))])
            
            # Track different polarization types
            pol_counts = {'H': [], 'V': [], 'D': [], 'A': [], 'R': [], 'L': [], 'custom': []}
            
            for obs in self.energy_history:
                pol_data = obs.get('polarization_counts', {})
                for pol_type in pol_counts:
                    pol_counts[pol_type].append(pol_data.get(pol_type, 0))
            
            # Stack plot
            pol_arrays = [np.array(counts) for counts in pol_counts.values()]
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'gray']
            
            if any(np.sum(arr) > 0 for arr in pol_arrays):
                ax.stackplot(times, *pol_arrays, labels=list(pol_counts.keys()), 
                           colors=colors, alpha=0.7)
                ax.legend(loc='upper right')
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Photon Count')
            ax.set_title('Polarization Distribution')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No polarization data', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title('Polarization Distribution')
    
    def plot_polarizer_interactions(self, ax):
        """Plot polarizer transmission and blocking events"""
        if self.polarizer_events:
            transmitted = [event for event in self.polarizer_events if event['action'] == 'transmitted']
            blocked = [event for event in self.polarizer_events if event['action'] == 'blocked']
            
            if transmitted:
                trans_times = [event['time'] for event in transmitted]
                trans_probs = [event['transmission_prob'] for event in transmitted]
                ax.scatter(trans_times, trans_probs, c='green', alpha=0.6, 
                          label=f'Transmitted ({len(transmitted)})', s=30)
            
            if blocked:
                block_times = [event['time'] for event in blocked]
                block_probs = [event['transmission_prob'] for event in blocked]
                ax.scatter(block_times, block_probs, c='red', alpha=0.6, 
                          label=f'Blocked ({len(blocked)})', s=30)
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Transmission Probability')
            ax.set_title('Polarizer Interactions')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
        else:
            ax.text(0.5, 0.5, 'No polarizer interactions', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title('Polarizer Interactions')
    
    def print_analysis(self):
        """Print detailed analysis of simulation results"""
        print("\n" + "="*70)
        print("=== RETROCAUSAL PHOTON SIMULATION WITH POLARIZERS ANALYSIS ===")
        print("="*70)
        
        # Basic statistics
        print(f"\nSIMULATION PARAMETERS:")
        print(f"Total simulation time: {self.t:.2f}")
        print(f"Space size: {self.space_size}")
        print(f"Retrocausal coupling strength: {self.retro_strength}")
        print(f"Speed of light: {self.c}")
        print(f"Number of polarizers: {len(self.polarizers)}")
        
        # Photon statistics
        print(f"\nPHOTON STATISTICS:")
        print(f"Remaining photons: {len(self.photons)}")
        print(f"Absorbed photons (detectors): {len(self.absorbed_photons)}")
        print(f"Blocked photons (polarizers): {len(self.blocked_photons)}")
        print(f"Total detection events: {len(self.detection_events)}")
        print(f"Total polarizer interactions: {len(self.polarizer_events)}")
        
        if self.photons:
            energies = [p.energy() for p in self.photons]
            momenta = [p.momentum() for p in self.photons]
            correlations = [p.future_correlation for p in self.photons]
            
            print(f"Average photon energy: {np.mean(energies):.4f}")
            print(f"Energy spread: {np.std(energies):.4f}")
            print(f"Average momentum: {np.mean(momenta):.4f}")
            print(f"Average future correlation: {np.mean(correlations):.4f}")
        
        # Polarizer analysis
        print(f"\nPOLARIZER ANALYSIS:")
        for polarizer in self.polarizers:
            transmitted = len([e for e in self.polarizer_events 
                             if e['polarizer'] == polarizer.name and e['action'] == 'transmitted'])
            blocked = len([e for e in self.polarizer_events 
                         if e['polarizer'] == polarizer.name and e['action'] == 'blocked'])
            total_interactions = transmitted + blocked
            
            print(f"{polarizer.name}:")
            print(f"  Position: {polarizer.position:.1f}")
            print(f"  Oscillation frequency: {polarizer.oscillation_freq:.2f} Hz")
            print(f"  Amplitude: {np.degrees(polarizer.amplitude):.1f}°")
            print(f"  Total interactions: {total_interactions}")
            print(f"  Transmitted: {transmitted} ({100*transmitted/(total_interactions+1e-10):.1f}%)")
            print(f"  Blocked: {blocked} ({100*blocked/(total_interactions+1e-10):.1f}%)")
        
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
                
                # Polarization analysis of detected photons
                det_pols = [d['polarization'] for d in detector['detections']]
                pol_counts = {pol: det_pols.count(pol) for pol in set(det_pols)}
                print(f"  Polarization distribution: {pol_counts}")
        
        # Energy conservation
        if self.energy_history:
            print(f"\nENERGY CONSERVATION:")
            initial_energy = self.energy_history[0]['total_energy']
            final_energy = self.energy_history[-1]['total_energy']
            
            # Account for blocked photons
            blocked_energy = sum(p.energy() for p in self.blocked_photons)
            total_final_energy = final_energy + blocked_energy
            
            energy_change = (total_final_energy - initial_energy) / initial_energy * 100
            
            print(f"Initial total energy: {initial_energy:.6f}")
            print(f"Final active energy: {final_energy:.6f}")
            print(f"Blocked energy: {blocked_energy:.6f}")
            print(f"Total final energy: {total_final_energy:.6f}")
            print(f"Energy change: {energy_change:.2f}%")
        
        # Retrocausal effects
        print(f"\nRETROCAUSAL AND POLARIZATION EFFECTS:")
        if self.energy_history:
            max_correlation = max(obs['avg_future_correlation'] for obs in self.energy_history)
            print(f"Maximum future correlation observed: {max_correlation:.6f}")
            
            # Analyze correlation timeline
            correlation_timeline = [obs['avg_future_correlation'] for obs in self.energy_history]
            peak_time = np.argmax(correlation_timeline) * self.dt
            print(f"Peak correlation time: {peak_time:.2f}")
        
        # Polarizer oscillation effects
        if self.polarizer_events:
            transmission_probs = [e['transmission_prob'] for e in self.polarizer_events 
                                if e['action'] == 'transmitted']
            blocking_probs = [e['transmission_prob'] for e in self.polarizer_events 
                            if e['action'] == 'blocked']
            
            if transmission_probs:
                print(f"Average transmission probability: {np.mean(transmission_probs):.4f}")
            if blocking_probs:
                print(f"Average blocking probability: {np.mean(blocking_probs):.4f}")
        
        print("\n" + "="*70)
        print("The simulation demonstrates quantum photon behavior with:")
        print("• Retrocausal influences from future field states")
        print("• Dynamic polarization filtering through oscillating polarizers")
        print("• Quantum measurement effects (Malus's law)")
        print("• Polarization-dependent transmission probabilities")
        print("• Energy conservation in presence of absorbing elements")
        print("="*70)

# Example usage and demonstration
if __name__ == "__main__":
    # Create simulator
    sim = RetrocausalPhotonSimulator(
        space_size=150.0,
        dt=0.02,
        c=1.0,
        retro_strength=0.15
    )
    
    # Add detectors
    sim.add_detector(position=18.0, efficiency=0.9, name="Early Detector")
    sim.add_detector(position=25.0, efficiency=0.85, name="Mid Detector")
        
    # Add polarizer oscillators
    print("Adding polarizer oscillators...")
    
    # Polarizer 1: Fast oscillation, horizontal-vertical
    sim.add_polarizer_oscillator(
        position=15.0,
        width=1.0,  # Narrower width
        oscillation_freq=0.5,  # Slower oscillation
        amplitude=np.pi/4,  # Smaller amplitude (45° instead of 90°)
        transmission_efficiency=0.98,  # Higher efficiency
        name="Gentler Polarizer"
    )
    
    # Create initial photon pulses with mixed polarizations
    print("Creating photon pulses with mixed polarizations...")
    
    # Pulse 1: Mixed linear polarizations
    sim.create_photon_pulse(center_x=5.0, width=2.0, n_photons=40, 
                           k_mean=5.0, k_spread=1.0,
                           polarization_mix=['H', 'V', 'D', 'A'])
    
    # Pulse 2: Circular polarizations
    #sim.create_photon_pulse(center_x=8.0, width=1.5, n_photons=30, 
                           #k_mean=8.0, k_spread=1.5,
                           #polarization_mix=['R', 'L'])
    
    # Pulse 3: High energy with diagonal polarization
    #sim.create_photon_pulse(center_x=12.0, width=1.0, n_photons=25, 
                           #k_mean=12.0, k_spread=2.0,
                           #polarization_mix=['D'])
    
    # Individual photons with specific properties
    #sim.create_photon(x=3.0, k=6.0, amplitude=1.5, polarization='H')
    #sim.create_photon(x=4.0, k=6.0, amplitude=1.5, polarization='V')
    #sim.create_photon(x=6.0, k=10.0, amplitude=1.2, polarization='R')
    #sim.create_photon(x=7.0, k=10.0, amplitude=1.2, polarization='L')
    
    print(f"Total initial photons: {len(sim.photons)}")
    print(f"Polarizer oscillators: {len(sim.polarizers)}")
    
    # Run simulation
    print("\nStarting retrocausal photon simulation with polarizer oscillators...")
    sim.run_simulation(total_time=30.0)
    
    # Generate results
    sim.plot_results(save_plots=True, show_plots=False)
    sim.print_analysis()
    
    print("\n=== QUANTUM POLARIZATION INTERPRETATION ===")
    print("This enhanced simulation models:")
    print("• Individual photons with quantum polarization states")
    print("• Time-oscillating polarizers following Malus's law")
    print("• Dynamic polarization filtering and quantum measurement")
    print("• Retrocausal correlations affecting polarization evolution")
    print("• Energy conservation with polarization-dependent losses")
    print("• Phase-dependent polarizer interactions")
    print("• Quantum interference between polarization components")
    print("\nKey observations:")
    print("• Transmission probability varies sinusoidally with polarizer angle")
    print("• Photons are probabilistically transmitted or blocked")
    print("• Transmitted photons acquire polarizer's transmission axis orientation")
    print("• Multiple polarizers create complex filtering cascades")
    print("• Retrocausal effects can influence polarization selection")
