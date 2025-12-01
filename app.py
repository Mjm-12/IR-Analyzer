import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import io

# Page configuration
st.set_page_config(
    page_title="Guitar IR Analyzer",
    page_icon="🎸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve dark mode visibility and hide default footer
st.markdown("""
<style>
    /* Hide Streamlit footer */
    footer {visibility: hidden;}

    /* Ensure good contrast in dark mode */
    .stApp {
        color: inherit;
    }

    /* Style for file upload area */
    [data-testid="stFileUploadDropzone"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 2px dashed rgba(255, 255, 255, 0.3);
    }

    /* Success message styling */
    .success-message {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: rgba(0, 128, 0, 0.2);
        border: 1px solid rgba(0, 128, 0, 0.5);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def analyze_ir(audio_data, sample_rate, display_duration, fft_window_size, window_function='boxcar'):
    """
    Analyze impulse response and return waveform and FFT data

    Args:
        audio_data: Audio samples
        sample_rate: Sample rate in Hz
        display_duration: Duration to display in seconds
        fft_window_size: FFT window size
        window_function: Window function name for FFT

    Returns:
        tuple: (time_array, waveform, frequencies, magnitude_db)
    """
    # Ensure mono
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Normalize
    audio_data = audio_data / np.max(np.abs(audio_data))

    # Calculate samples to display
    samples_to_display = int(display_duration * sample_rate)
    samples_to_display = min(samples_to_display, len(audio_data))

    # Time array for waveform
    time_array = np.arange(samples_to_display) / sample_rate * 1000  # Convert to ms
    waveform = audio_data[:samples_to_display]

    # Apply window function for FFT
    if window_function == 'boxcar':
        window = signal.windows.boxcar(len(audio_data))
    elif window_function == 'hann':
        window = signal.windows.hann(len(audio_data))
    elif window_function == 'hamming':
        window = signal.windows.hamming(len(audio_data))
    elif window_function == 'blackman':
        window = signal.windows.blackman(len(audio_data))
    elif window_function == 'bartlett':
        window = signal.windows.bartlett(len(audio_data))
    elif window_function == 'kaiser':
        window = signal.windows.kaiser(len(audio_data), beta=8.6)
    elif window_function == 'blackmanharris':
        window = signal.windows.blackmanharris(len(audio_data))
    elif window_function == 'flattop':
        window = signal.windows.flattop(len(audio_data))
    elif window_function == 'tukey':
        window = signal.windows.tukey(len(audio_data))
    else:
        window = signal.windows.boxcar(len(audio_data))

    # Apply window to audio data
    windowed_data = audio_data * window

    # FFT Analysis
    fft_result = np.fft.rfft(windowed_data, n=fft_window_size)
    magnitude = np.abs(fft_result)

    # Convert to dB (avoid log of zero)
    magnitude_db = 20 * np.log10(magnitude + 1e-10)

    # Frequency array
    frequencies = np.fft.rfftfreq(fft_window_size, 1/sample_rate)

    return time_array, waveform, frequencies, magnitude_db

def plot_waveforms(waveform_data, filenames, display_duration_ms):
    """
    Plot all waveforms in a single graph with dark mode support

    Args:
        waveform_data: List of (time_array, waveform) tuples
        filenames: List of filenames
        display_duration_ms: Display duration in milliseconds
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for idx, ((time_ms, waveform), filename) in enumerate(zip(waveform_data, filenames)):
        color = colors[idx % len(colors)]
        ax.plot(time_ms, waveform, linewidth=0.8, color=color, alpha=0.7, label=filename)

    ax.set_xlabel('Time (ms)', fontsize=10)
    ax.set_ylabel('Amplitude (Normalized)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)

    # Set x-axis range to 0 ~ display_duration_ms
    ax.set_xlim([0, display_duration_ms])

    # Set background to transparent for better theme compatibility
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # Set axis colors based on Streamlit theme
    ax.spines['bottom'].set_color('#888888')
    ax.spines['top'].set_color('#888888')
    ax.spines['left'].set_color('#888888')
    ax.spines['right'].set_color('#888888')
    ax.tick_params(colors='#888888')
    ax.xaxis.label.set_color('#888888')
    ax.yaxis.label.set_color('#888888')

    plt.tight_layout()
    return fig

def apply_octave_smoothing(frequencies, magnitude_db, octave_fraction):
    """
    Apply fractional octave smoothing to frequency response

    Args:
        frequencies: Frequency array
        magnitude_db: Magnitude in dB
        octave_fraction: Fraction of octave (e.g., 24 for 1/24 octave)

    Returns:
        Smoothed magnitude in dB
    """
    if octave_fraction == 0:  # No smoothing
        return magnitude_db

    smoothed = np.zeros_like(magnitude_db)

    for i, fc in enumerate(frequencies):
        if fc <= 0:
            smoothed[i] = magnitude_db[i]
            continue

        # Calculate frequency band based on octave fraction
        f_lower = fc / (2 ** (1 / (2 * octave_fraction)))
        f_upper = fc * (2 ** (1 / (2 * octave_fraction)))

        # Find indices within the band
        mask = (frequencies >= f_lower) & (frequencies <= f_upper)

        if np.any(mask):
            # Average in linear scale then convert back to dB
            linear_avg = np.mean(10 ** (magnitude_db[mask] / 20))
            smoothed[i] = 20 * np.log10(linear_avg + 1e-10)
        else:
            smoothed[i] = magnitude_db[i]

    return smoothed

def plot_fft(frequencies, magnitude_db, filenames, octave_smoothing=0):
    """
    Plot FFT frequency response with dark mode support

    Args:
        frequencies: List of frequency arrays
        magnitude_db: List of magnitude arrays in dB
        filenames: List of filenames
        octave_smoothing: Octave fraction for smoothing (0 = no smoothing)
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for idx, (freq, mag_db, filename) in enumerate(zip(frequencies, magnitude_db, filenames)):
        # Apply octave smoothing if requested
        if octave_smoothing > 0:
            mag_db = apply_octave_smoothing(freq, mag_db, octave_smoothing)

        # Normalize each file to 0dB max
        max_magnitude = np.max(mag_db)
        mag_db_normalized = mag_db - max_magnitude

        color = colors[idx % len(colors)]
        ax.semilogx(freq, mag_db_normalized, label=filename, linewidth=1.5,
                   color=color, alpha=0.8)

    ax.set_xlabel('Frequency (Hz)', fontsize=10)
    ax.set_ylabel('Magnitude (dB, Normalized)', fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='best', fontsize=9)

    # Set reasonable frequency limits
    ax.set_xlim([20, 20000])

    # Set x-axis ticks to engineering format: 20, 100, 1k, 10k, 20k
    ax.set_xticks([20, 100, 1000, 10000, 20000])
    ax.set_xticklabels(['20', '100', '1k', '10k', '20k'])

    # Set background to transparent for better theme compatibility
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # Set axis colors
    ax.spines['bottom'].set_color('#888888')
    ax.spines['top'].set_color('#888888')
    ax.spines['left'].set_color('#888888')
    ax.spines['right'].set_color('#888888')
    ax.tick_params(colors='#888888')
    ax.xaxis.label.set_color('#888888')
    ax.yaxis.label.set_color('#888888')

    plt.tight_layout()
    return fig

# Main App
st.title("Guitar IR Analyzer")
st.markdown("Guitar Cabinet Impulse Response Analysis Tool")
st.markdown("Upload multiple WAV files to analyze and visualize waveforms and FFT results.")

# Sidebar for settings
st.sidebar.header("Analysis Settings")

# Display duration setting (in milliseconds)
display_duration_ms = st.sidebar.slider(
    "Waveform Display Duration (ms)",
    min_value=1,
    max_value=100,
    value=10,
    step=1,
    help="Duration of waveform to display in the time domain plot"
)
# Convert to seconds for processing
display_duration = display_duration_ms / 1000.0

# FFT settings
fft_window_size = st.sidebar.selectbox(
    "FFT Window Size",
    options=[2048, 4096, 8192, 16384, 32768, 65536],
    index=3,
    help="Larger window size provides better frequency resolution"
)

# Window function selection
window_function = st.sidebar.selectbox(
    "FFT Window Function",
    options=[
        "Rectangular (Boxcar)",
        "Hann",
        "Hamming",
        "Blackman",
        "Bartlett",
        "Kaiser",
        "Blackman-Harris",
        "Flat Top",
        "Tukey"
    ],
    index=0,
    help="Window function applied before FFT analysis"
)
# Map display names to function names
window_map = {
    "Rectangular (Boxcar)": "boxcar",
    "Hann": "hann",
    "Hamming": "hamming",
    "Blackman": "blackman",
    "Bartlett": "bartlett",
    "Kaiser": "kaiser",
    "Blackman-Harris": "blackmanharris",
    "Flat Top": "flattop",
    "Tukey": "tukey"
}
window_func_name = window_map[window_function]

# Octave smoothing
smoothing_options = {
    "None": 0,
    "1/48 Octave": 48,
    "1/24 Octave": 24,
    "1/12 Octave": 12,
    "1/6 Octave": 6,
    "1/3 Octave": 3,
    "1 Octave": 1
}
smoothing_label = st.sidebar.selectbox(
    "FFT Smoothing",
    options=list(smoothing_options.keys()),
    index=2,  # Default to 1/24 Octave
    help="Apply fractional octave smoothing to FFT results"
)
smoothing = smoothing_options[smoothing_label]

# File Upload Section
st.header("File Upload")

uploaded_files = st.file_uploader(
    "Drag and drop files here",
    type=['wav'],
    accept_multiple_files=True,
    help="Limit 200MB per file - WAV"
)

# Display file list
if uploaded_files:
    st.markdown(f"**{len(uploaded_files)} file(s) uploaded**")
    for i, file in enumerate(uploaded_files, 1):
        st.text(f"  {i}. {file.name}")

# Add execution button
analyze_button = st.button("Start Analysis", type="primary", disabled=(not uploaded_files))

# Analysis section - only runs when button is clicked
if analyze_button and uploaded_files:
    st.success(f"Processing {len(uploaded_files)} file(s)...")

    # Store analysis results
    waveform_data = []
    fft_data = []
    filenames = []

    # Progress bar
    progress_bar = st.progress(0)

    for idx, uploaded_file in enumerate(uploaded_files):
        try:
            # Read WAV file
            sample_rate, audio_data = wavfile.read(io.BytesIO(uploaded_file.read()))

            # Analyze
            time_array, waveform, frequencies, magnitude_db = analyze_ir(
                audio_data,
                sample_rate,
                display_duration,
                fft_window_size,
                window_func_name
            )

            waveform_data.append((time_array, waveform))
            fft_data.append((frequencies, magnitude_db))
            filenames.append(uploaded_file.name)

            # Update progress
            progress_bar.progress((idx + 1) / len(uploaded_files))

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")

    # Clear progress bar
    progress_bar.empty()

    if waveform_data:
        # Display waveforms in a single graph
        st.header("Waveform (Time Domain)")
        fig = plot_waveforms(waveform_data, filenames, display_duration_ms)
        st.pyplot(fig)
        plt.close(fig)

        # Display FFT analysis
        st.header("FFT Analysis (Frequency Response)")

        # Create combined FFT plot
        all_frequencies = [data[0] for data in fft_data]
        all_magnitudes = [data[1] for data in fft_data]

        fig = plot_fft(all_frequencies, all_magnitudes, filenames, smoothing)
        st.pyplot(fig)
        plt.close(fig)

        st.success("Analysis complete!")
    else:
        st.warning("No files were successfully processed.")
elif not uploaded_files:
    st.info("Please upload WAV files to begin analysis.")
