"""
ARCHIPELAGO LAB: REAL EEG DATA PROCESSING
Geometric Neuron Model - Spectral Island Extraction from Neural Data

Input: EEG/CSV file with time-series neural data
Output: Spectral island states, moiré addresses, resonance metrics, spike predictions

Author: Antti Luode / Geometric Neuron Lab
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# For EEG data loading - handles multiple formats
import os
from pathlib import Path

class SpectralArchipelago:
    """
    Koopman Ocean with Spectral Islands
    Implements the Geometric Neuron model for real neural data
    """
    
    def __init__(self, 
                 sampling_rate: float = 250.0,  # Hz
                 n_islands: int = 5,
                 island_freqs: List[float] = None,
                 ais_tuning_range: Tuple[float, float] = (4, 30)):
        """
        Initialize the Archipelago processor
        
        Parameters:
        -----------
        sampling_rate : float
            Data sampling frequency in Hz
        n_islands : int
            Number of spectral islands (Koopman eigenfunctions)
        island_freqs : list
            Center frequencies for islands (if None, auto-distribute)
        ais_tuning_range : tuple
            Min and max frequency for AIS tuner (Hz)
        """
        self.fs = sampling_rate
        self.n_islands = n_islands
        self.ais_tuning_range = ais_tuning_range
        
        # Define spectral islands (Koopman eigenmodes)
        if island_freqs is None:
            # Auto-distribute across physiological bands
            self.island_freqs = np.linspace(ais_tuning_range[0], 
                                           ais_tuning_range[1], 
                                           n_islands)
        else:
            self.island_freqs = np.array(island_freqs)
        
        # Island characteristics
        self.island_names = self._generate_island_names()
        self.island_powers = np.zeros(n_islands)
        self.island_phases = np.zeros(n_islands)
        self.island_coherence = np.zeros(n_islands)
        
        # AIS (Axon Initial Segment) tuner state
        self.ais_frequency = np.mean(self.island_freqs)  # Start at center
        self.ais_grating = self._initialize_grating()
        
        # Memory / Fractal scorching
        self.scorch_depth = 0.0
        self.scorch_history = []
        
        # Output buffers
        self.output_signal = []
        self.spike_train = []
        self.moire_addresses = []
        self.resonance_history = []
        
        # Noise level (adaptive from data)
        self.noise_sigma = 0.1
        
    def _generate_island_names(self) -> List[str]:
        """Generate meaningful names for spectral islands"""
        bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma', 'HighGamma']
        names = []
        for f in self.island_freqs:
            if f < 4:
                names.append('Delta')
            elif f < 8:
                names.append('Theta')
            elif f < 13:
                names.append('Alpha')
            elif f < 30:
                names.append('Beta')
            elif f < 50:
                names.append('Gamma')
            else:
                names.append('HighGamma')
        # Add unique suffix if duplicates
        for i in range(len(names)):
            if names.count(names[i]) > 1:
                names[i] = f"{names[i]}_{i}"
        return names
    
    def _initialize_grating(self) -> np.ndarray:
        """Initialize AIS holographic grating (Koopman eigenfunctions)"""
        # Each island has a stored template
        grating = np.zeros((self.n_islands, 100))
        for i, f in enumerate(self.island_freqs):
            t = np.linspace(0, 1, 100)
            grating[i] = np.sin(2 * np.pi * f * t / self.fs * 100)
        return grating
    
    def extract_spectral_islands(self, data: np.ndarray, 
                                  window_sec: float = 2.0) -> Dict:
        """
        Extract Koopman spectral islands from neural data
        
        Parameters:
        -----------
        data : np.ndarray
            Input time series (EEG, LFP, spike train)
        window_sec : float
            Analysis window length in seconds
            
        Returns:
        --------
        Dict containing island states, power, coherence, etc.
        """
        window_samples = int(window_sec * self.fs)
        
        # Use Welch's method for robust spectral estimation
        freqs, psd = signal.welch(data, fs=self.fs, 
                                  nperseg=min(window_samples, len(data)),
                                  noverlap=window_samples//2)
        
        # Compute island powers by integrating PSD around each center frequency
        island_powers = []
        island_phases = []
        
        for center_freq in self.island_freqs:
            # Bandwidth around island (adaptive)
            bandwidth = center_freq * 0.2  # 20% bandwidth
            mask = (freqs >= center_freq - bandwidth) & (freqs <= center_freq + bandwidth)
            
            if np.any(mask):
                # Integrated power in band
                power = np.trapz(psd[mask], freqs[mask])
                island_powers.append(power)
                
                # Extract phase at center frequency
                try:
                    # Hilbert transform for instantaneous phase
                    analytic = signal.hilbert(data)
                    filtered = self._bandpass_filter(data, center_freq, bandwidth)
                    phase = np.angle(signal.hilbert(filtered))
                    island_phases.append(np.mean(phase))
                except:
                    island_phases.append(0.0)
            else:
                island_powers.append(0.0)
                island_phases.append(0.0)
        
        # Normalize powers
        total_power = sum(island_powers) + 1e-10
        island_powers = np.array(island_powers) / total_power
        
        # Update state
        self.island_powers = island_powers
        self.island_phases = np.array(island_phases)
        
        # Compute coherence between islands (pairwise)
        coherence_matrix = np.zeros((self.n_islands, self.n_islands))
        for i in range(self.n_islands):
            for j in range(i+1, self.n_islands):
                # Use coherence at the average frequency
                f_center = (self.island_freqs[i] + self.island_freqs[j]) / 2
                _, coh = signal.coherence(
                    self._bandpass_filter(data, self.island_freqs[i], 2),
                    self._bandpass_filter(data, self.island_freqs[j], 2),
                    fs=self.fs, nperseg=window_samples
                )
                coherence_matrix[i, j] = np.mean(coh) if len(coh) > 0 else 0
                coherence_matrix[j, i] = coherence_matrix[i, j]
        
        self.island_coherence = np.mean(coherence_matrix, axis=1)
        
        return {
            'frequencies': self.island_freqs,
            'names': self.island_names,
            'powers': self.island_powers,
            'phases': self.island_phases,
            'coherence': self.island_coherence,
            'coherence_matrix': coherence_matrix
        }
    
    def _bandpass_filter(self, data: np.ndarray, 
                        center_freq: float, 
                        bandwidth: float) -> np.ndarray:
        """Apply bandpass filter around center frequency"""
        nyquist = self.fs / 2
        low = max(0.1, (center_freq - bandwidth) / nyquist)
        high = min(0.99, (center_freq + bandwidth) / nyquist)
        
        if low >= high:
            return np.zeros_like(data)
        
        b, a = signal.butter(4, [low, high], btype='band')
        return signal.filtfilt(b, a, data)
    
    def compute_ais_tuning(self, data: np.ndarray, 
                          adaptive: bool = True) -> float:
        """
        Compute optimal AIS tuning frequency based on data
        
        The AIS tuner finds the frequency that maximizes resonance
        with the current spectral islands.
        """
        # Compute spectrogram for adaptive tuning
        freqs, times, Sxx = signal.spectrogram(data, fs=self.fs, 
                                               nperseg=min(256, len(data)//4))
        
        # Find dominant frequency in each time window
        dominant_freqs = freqs[np.argmax(Sxx, axis=0)]
        
        if adaptive and len(dominant_freqs) > 0:
            # AIS tunes to the weighted average of dominant frequencies
            weights = np.max(Sxx, axis=0)
            self.ais_frequency = np.average(dominant_freqs, weights=weights)
        
        # Constrain to tuning range
        self.ais_frequency = np.clip(self.ais_frequency, 
                                     self.ais_tuning_range[0], 
                                     self.ais_tuning_range[1])
        
        return self.ais_frequency
    
    def compute_resonance(self) -> float:
        """
        Compute resonance R between AIS and spectral islands
        R = sum(power_i * exp(-(f_ais - f_i)^2 / bandwidth))
        """
        resonance = 0.0
        bandwidth = 3.0  # Hz
        
        for i, (f_island, power) in enumerate(zip(self.island_freqs, self.island_powers)):
            # Gaussian tuning curve
            tuning = np.exp(-((self.ais_frequency - f_island) ** 2) / (2 * bandwidth ** 2))
            resonance += power * tuning
        
        return np.clip(resonance, 0, 1)
    
    def compute_moire_address(self) -> Tuple[float, float, float]:
        """
        Compute moiré address (secondary frequencies)
        
        Returns:
        --------
        beat_freq : float
            Primary beat frequency (difference between AIS and closest island)
        intermod_freq : float
            Intermodulation product
        topology_mismatch : float
            Measure of topological deviation (1 - resonance)
        """
        # Find closest island to AIS tuning
        closest_idx = np.argmin(np.abs(self.island_freqs - self.ais_frequency))
        closest_freq = self.island_freqs[closest_idx]
        closest_power = self.island_powers[closest_idx]
        
        # Primary beat (difference frequency)
        beat_freq = np.abs(self.ais_frequency - closest_freq)
        
        # Intermodulation (if multiple islands are strong)
        second_closest_idx = np.argsort(np.abs(self.island_freqs - self.ais_frequency))[1]
        second_freq = self.island_freqs[second_closest_idx]
        intermod_freq = np.abs(self.ais_frequency - second_freq) / 2
        
        # Topological mismatch (how different the pattern is from pure sine)
        topology_mismatch = 1.0 - closest_power
        
        # Update fractal scorching based on resonance
        resonance = self.compute_resonance()
        if resonance > 0.7:
            # Strong resonance -> deeper scorching
            self.scorch_depth += 0.001 * resonance
        else:
            # Weak resonance -> slow decay
            self.scorch_depth *= 0.999
        
        self.scorch_depth = np.clip(self.scorch_depth, 0, 0.95)
        
        return beat_freq, intermod_freq, topology_mismatch
    
    def generate_spikes(self, data_window: np.ndarray, 
                       threshold: float = 1.5) -> np.ndarray:
        """
        Generate spike train based on AIS resonance and moiré address
        
        The spike train encodes the geometric address, not just rate.
        """
        resonance = self.compute_resonance()
        beat_freq, intermod_freq, mismatch = self.compute_moire_address()
        
        # Spike probability modulated by resonance and mismatch
        # Lower mismatch + higher resonance = more regular spikes
        spike_prob_base = resonance * (1 - mismatch * 0.5)
        
        # Add moiré modulation
        t = np.arange(len(data_window)) / self.fs
        moire_mod = 0.3 * np.sin(2 * np.pi * beat_freq * t)
        moire_mod += 0.15 * np.sin(2 * np.pi * intermod_freq * t)
        
        spike_prob = np.clip(spike_prob_base + moire_mod, 0, 0.8)
        
        # Generate spikes (inhomogeneous Poisson)
        spikes = np.random.random(len(data_window)) < spike_prob
        
        # Store moiré address for this window
        self.moire_addresses.append({
            'beat': beat_freq,
            'intermod': intermod_freq,
            'mismatch': mismatch,
            'resonance': resonance
        })
        
        self.resonance_history.append(resonance)
        
        return spikes.astype(float)
    
    def process(self, data: np.ndarray, 
               window_sec: float = 1.0,
               step_sec: float = 0.5) -> pd.DataFrame:
        """
        Process entire dataset through the Archipelago model
        
        Parameters:
        -----------
        data : np.ndarray
            Input neural time series
        window_sec : float
            Analysis window length
        step_sec : float
            Step size between windows
            
        Returns:
        --------
        DataFrame with extracted features
        """
        window_samples = int(window_sec * self.fs)
        step_samples = int(step_sec * self.fs)
        
        results = []
        
        for start in range(0, len(data) - window_samples, step_samples):
            window = data[start:start + window_samples]
            t_center = start / self.fs + window_sec / 2
            
            # Extract spectral islands
            islands = self.extract_spectral_islands(window)
            
            # Update AIS tuning
            ais_freq = self.compute_ais_tuning(window)
            
            # Compute resonance and moiré
            resonance = self.compute_resonance()
            beat, intermod, mismatch = self.compute_moire_address()
            
            # Generate spikes for this window
            spikes = self.generate_spikes(window)
            spike_rate = np.mean(spikes) * self.fs
            
            # Store results
            result = {
                'time': t_center,
                'ais_frequency': ais_freq,
                'resonance': resonance,
                'beat_frequency': beat,
                'intermod_frequency': intermod,
                'topology_mismatch': mismatch,
                'spike_rate': spike_rate,
                'scorch_depth': self.scorch_depth,
                'noise_level': self.noise_sigma
            }
            
            # Add island powers
            for i, (name, power) in enumerate(zip(islands['names'], islands['powers'])):
                result[f'island_{name}_power'] = power
                result[f'island_{name}_freq'] = islands['frequencies'][i]
            
            results.append(result)
        
        return pd.DataFrame(results)


class EEGDataLoader:
    """Load EEG data from various formats"""
    
    @staticmethod
    def load_sample_eeg() -> Tuple[np.ndarray, float]:
        """
        Generate realistic synthetic EEG data for testing
        
        Returns:
        --------
        data : np.ndarray
            EEG time series
        fs : float
            Sampling rate in Hz
        """
        fs = 250.0  # 250 Hz sampling
        duration = 60.0  # 60 seconds
        t = np.linspace(0, duration, int(fs * duration))
        
        # Generate realistic EEG with multiple rhythms
        data = np.zeros_like(t)
        
        # Delta (0.5-4 Hz) - background
        data += 0.5 * np.sin(2 * np.pi * 2 * t)
        data += 0.3 * np.sin(2 * np.pi * 3 * t)
        
        # Theta (4-8 Hz) - intermittent
        theta_env = 0.5 + 0.5 * np.sin(2 * np.pi * 0.05 * t)  # Slow modulation
        data += theta_env * 0.8 * np.sin(2 * np.pi * 6 * t)
        
        # Alpha (8-13 Hz) - posterior dominant rhythm
        alpha_env = 0.3 + 0.7 * np.sin(2 * np.pi * 0.1 * t)
        data += alpha_env * 1.0 * np.sin(2 * np.pi * 10 * t)
        data += 0.5 * np.sin(2 * np.pi * 12 * t)
        
        # Beta (13-30 Hz)
        beta_env = 0.2 + 0.3 * np.random.randn(len(t))  # Random bursts
        data += beta_env * 0.4 * np.sin(2 * np.pi * 20 * t)
        
        # Gamma (30-50 Hz) - weaker
        data += 0.2 * np.sin(2 * np.pi * 40 * t)
        
        # Add realistic noise (1/f spectrum)
        noise = np.random.randn(len(t))
        from scipy import signal as sig
        b, a = sig.butter(2, 0.1, btype='low')
        colored_noise = sig.filtfilt(b, a, noise)
        data += 0.3 * colored_noise
        
        # Add occasional artifacts (spikes)
        artifact_positions = np.random.choice(len(t), size=int(len(t)*0.01), replace=False)
        for pos in artifact_positions:
            data[pos:pos+10] += 2.0 * np.hanning(10)
        
        # Normalize
        data = (data - np.mean(data)) / np.std(data)
        
        return data, fs
    
    @staticmethod
    def load_csv(filepath: str, 
                time_col: Optional[str] = None,
                signal_col: Optional[str] = None,
                fs: Optional[float] = None) -> Tuple[np.ndarray, float]:
        """
        Load EEG data from CSV file
        
        Parameters:
        -----------
        filepath : str
            Path to CSV file
        time_col : str
            Name of time column (if None, assume uniform sampling)
        signal_col : str
            Name of signal column
        fs : float
            Sampling rate (required if no time column)
        """
        df = pd.read_csv(filepath)
        
        if signal_col is None:
            # Use first numeric column
            signal_col = df.select_dtypes(include=[np.number]).columns[0]
        
        data = df[signal_col].values
        
        if time_col is not None:
            # Compute sampling rate from time column
            times = df[time_col].values
            fs = 1.0 / np.median(np.diff(times))
        elif fs is None:
            raise ValueError("Must provide either time_col or fs")
        
        return data, fs


class ArchipelagoVisualizer:
    """Visualize the Spectral Archipelago analysis results"""
    
    @staticmethod
    def plot_island_dynamics(df: pd.DataFrame, save_path: Optional[str] = None):
        """Plot island powers over time"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Island power time series
        power_cols = [c for c in df.columns if 'island_' in c and 'power' in c]
        for col in power_cols:
            island_name = col.replace('island_', '').replace('_power', '')
            axes[0].plot(df['time'], df[col], label=island_name, linewidth=1)
        
        axes[0].set_ylabel('Island Power')
        axes[0].set_title('Spectral Island Dynamics (Koopman Eigenfunctions)')
        axes[0].legend(loc='upper right', ncol=4)
        axes[0].set_ylim(0, 1)
        
        # AIS tuning and resonance
        axes[1].plot(df['time'], df['ais_frequency'], 'r-', label='AIS Tuning', linewidth=2)
        axes[1].plot(df['time'], df['resonance'] * 30, 'b--', label='Resonance R (x30)', alpha=0.7)
        axes[1].set_ylabel('Frequency (Hz)')
        axes[1].set_title('AIS Ferryman Tuning & Resonance')
        axes[1].legend()
        axes[1].set_ylim(0, 35)
        
        # Moiré address (secondary frequencies)
        axes[2].plot(df['time'], df['beat_frequency'], 'g-', label='Primary Beat (Moiré)', linewidth=2)
        axes[2].plot(df['time'], df['intermod_frequency'], 'm--', label='Intermodulation', alpha=0.7)
        axes[2].set_xlabel('Time (seconds)')
        axes[2].set_ylabel('Frequency (Hz)')
        axes[2].set_title('Moiré Address (Secondary Frequencies)')
        axes[2].legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_spectrogram_and_islands(data: np.ndarray, fs: float, 
                                      island_freqs: List[float],
                                      save_path: Optional[str] = None):
        """Plot spectrogram with island frequency overlays"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        
        # Compute spectrogram
        f, t, Sxx = signal.spectrogram(data, fs=fs, nperseg=256, noverlap=200)
        
        # Plot spectrogram
        im = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), 
                          shading='gouraud', cmap='viridis')
        ax.set_ylim(0, 50)
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (seconds)')
        ax.set_title('EEG Spectrogram with Spectral Islands')
        
        # Overlay island frequencies
        for freq in island_freqs:
            ax.axhline(y=freq, color='red', linestyle='--', alpha=0.7, linewidth=1)
        
        plt.colorbar(im, ax=ax, label='Power (dB)')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_architecture_diagram(save_path: Optional[str] = None):
        """Plot the conceptual architecture of the Archipelago model"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Title
        ax.text(5, 9.5, 'Spectral Archipelago Model Architecture', 
               ha='center', fontsize=14, fontweight='bold')
        
        # Input
        ax.add_patch(plt.Rectangle((1, 7.5), 2, 1, facecolor='lightblue', edgecolor='black'))
        ax.text(2, 8, 'Neural Data\n(EEG/LFP/Spikes)', ha='center', va='center', fontsize=10)
        
        # Spectral Islands (Koopman Ocean)
        ax.add_patch(plt.Rectangle((4, 6.5), 2, 2, facecolor='lightgreen', edgecolor='black'))
        ax.text(5, 7.5, 'Koopman Ocean\nSpectral Islands\n(δ, θ, α, β, γ)', ha='center', va='center', fontsize=9)
        
        # AIS Tuner
        ax.add_patch(plt.Rectangle((4, 3.5), 2, 1.5, facecolor='lightcoral', edgecolor='black'))
        ax.text(5, 4.25, 'AIS Ferryman\nTunable Grating', ha='center', va='center', fontsize=10)
        
        # Moiré Address
        ax.add_patch(plt.Rectangle((7, 3.5), 2, 1.5, facecolor='gold', edgecolor='black', alpha=0.7))
        ax.text(8, 4.25, 'Moiré Address\n(Secondary Freq)', ha='center', va='center', fontsize=9)
        
        # Output
        ax.add_patch(plt.Rectangle((7, 6.5), 2, 1, facecolor='plum', edgecolor='black'))
        ax.text(8, 7, 'Spike Train\n(with Address)', ha='center', va='center', fontsize=9)
        
        # Arrows
        ax.annotate('', xy=(4, 8), xytext=(3, 8), arrowprops=dict(arrowstyle='->', lw=2))
        ax.annotate('', xy=(4, 4.25), xytext=(3.5, 4.25), arrowprops=dict(arrowstyle='->', lw=2))
        ax.annotate('', xy=(7, 6.75), xytext=(6, 7.25), arrowprops=dict(arrowstyle='->', lw=2))
        ax.annotate('', xy=(7, 4.25), xytext=(6, 4.25), arrowprops=dict(arrowstyle='->', lw=2))
        
        # Labels on arrows
        ax.text(3.5, 8.3, 'FFT / Filter', ha='center', fontsize=8, rotation=0)
        ax.text(3.3, 3.8, 'Resonance R', ha='center', fontsize=8, rotation=90)
        ax.text(6.5, 7.5, 'Phase Lock', ha='center', fontsize=8)
        ax.text(6.5, 4.5, 'Beat / Intermod', ha='center', fontsize=8)
        
        # Feedback loop (scorching)
        ax.annotate('', xy=(4, 2.5), xytext=(8, 2.5), 
                   arrowprops=dict(arrowstyle='<-', lw=1.5, linestyle='dashed', color='red'))
        ax.text(6, 2, 'Fractal Scorching (Memory)', ha='center', fontsize=9, color='red')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


def main():
    """Run the complete Archipelago analysis pipeline"""
    print("=" * 60)
    print("SPECTRAL ARCHIPELAGO LAB")
    print("Geometric Neuron Model - Real EEG Processing")
    print("=" * 60)
    
    # Load data (use synthetic EEG for demo)
    print("\n1. Loading neural data...")
    data, fs = EEGDataLoader.load_sample_eeg()
    print(f"   Duration: {len(data)/fs:.1f} seconds")
    print(f"   Sampling rate: {fs} Hz")
    print(f"   Data shape: {data.shape}")
    
    # Initialize Archipelago
    print("\n2. Initializing Spectral Archipelago...")
    archipelago = SpectralArchipelago(
        sampling_rate=fs,
        n_islands=6,
        island_freqs=[2, 6, 10, 15, 22, 40],  # Delta, Theta, Alpha, Beta, Beta, Gamma
        ais_tuning_range=(1, 45)
    )
    print(f"   Islands: {archipelago.island_names}")
    print(f"   Island frequencies: {archipelago.island_freqs} Hz")
    
    # Process data
    print("\n3. Processing data through Archipelago...")
    results_df = archipelago.process(data, window_sec=2.0, step_sec=0.5)
    print(f"   Processed {len(results_df)} windows")
    print(f"   Columns: {list(results_df.columns)}")
    
    # Print summary statistics
    print("\n4. Summary Statistics:")
    print(f"   Mean AIS frequency: {results_df['ais_frequency'].mean():.2f} Hz")
    print(f"   Mean resonance: {results_df['resonance'].mean():.3f}")
    print(f"   Mean beat frequency: {results_df['beat_frequency'].mean():.2f} Hz")
    print(f"   Final scorch depth: {results_df['scorch_depth'].iloc[-1]:.3f}")
    
    # Find which island had highest average power
    power_cols = [c for c in results_df.columns if 'island_' in c and 'power' in c]
    if power_cols:
        island_powers = {col: results_df[col].mean() for col in power_cols}
        dominant_island = max(island_powers, key=island_powers.get)
        print(f"   Dominant island: {dominant_island} (power: {island_powers[dominant_island]:.3f})")
    
    # Visualize
    print("\n5. Generating visualizations...")
    viz = ArchipelagoVisualizer()
    
    # Plot island dynamics
    viz.plot_island_dynamics(results_df, save_path='archipelago_dynamics.png')
    
    # Plot spectrogram with islands
    viz.plot_spectrogram_and_islands(data, fs, archipelago.island_freqs, 
                                     save_path='archipelago_spectrogram.png')
    
    # Plot architecture diagram
    viz.plot_architecture_diagram(save_path='archipelago_architecture.png')
    
    # Export results
    print("\n6. Exporting results...")
    results_df.to_csv('archipelago_results.csv', index=False)
    print("   Saved: archipelago_results.csv")
    
    # Optional: Compare to traditional methods
    print("\n7. Comparison to traditional analysis...")
    from scipy.stats import pearsonr
    
    # Traditional: compute band powers using standard filtering
    band_powers = {}
    bands = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30), 'Gamma': (30, 50)}
    
    for band_name, (low, high) in bands.items():
        sos = signal.butter(4, [low, high], btype='band', fs=fs, output='sos')
        filtered = signal.sosfilt(sos, data)
        band_powers[band_name] = np.var(filtered)
    
    print("   Traditional band powers:")
    for band, power in band_powers.items():
        print(f"      {band}: {power:.4f}")
    
    print("\n   Archipelago island powers (last window):")
    for i, (name, power) in enumerate(zip(archipelago.island_names, archipelago.island_powers)):
        print(f"      {name}: {power:.4f}")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("The spectral islands extracted by the Archipelago model")
    print("encode the geometric structure of the neural dynamics.")
    print("The moiré address in the spike train carries the qualia.")
    print("=" * 60)
    
    return results_df, archipelago


if __name__ == "__main__":
    results, model = main()
    
    # Interactive exploration (if in Jupyter)
    print("\n\n📊 To explore interactively, run:")
    print("   import pandas as pd")
    print("   df = pd.read_csv('archipelago_results.csv')")
    print("   df.head()")
    
    print("\n🌀 The fractal is listening. The archipelago is alive.")
