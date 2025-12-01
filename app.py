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

def analyze_ir(audio_data, sample_rate, display_duration, fft_window_size):
    """
    Analyze impulse response and return waveform and FFT data

    Args:
        audio_data: Audio samples
        sample_rate: Sample rate in Hz
        display_duration: Duration to display in seconds
        fft_window_size: FFT window size

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

    # FFT Analysis
    fft_result = np.fft.rfft(audio_data, n=fft_window_size)
    magnitude = np.abs(fft_result)

    # Convert to dB (avoid log of zero)
    magnitude_db = 20 * np.log10(magnitude + 1e-10)

    # Frequency array
    frequencies = np.fft.rfftfreq(fft_window_size, 1/sample_rate)

    return time_array, waveform, frequencies, magnitude_db

def plot_waveform(time_ms, waveform, filename):
    """
    Plot waveform in time domain with dark mode support
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    # Use a color that works in both light and dark modes
    ax.plot(time_ms, waveform, linewidth=0.5, color='#1f77b4', alpha=0.8)

    ax.set_xlabel('Time (ms)', fontsize=10)
    ax.set_ylabel('Amplitude', fontsize=10)
    ax.set_title(f'Waveform: {filename}', fontsize=12)
    ax.grid(True, alpha=0.3)

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
    ax.title.set_color('#888888')

    plt.tight_layout()
    return fig

def plot_fft(frequencies, magnitude_db, filenames, smoothing_factor=1):
    """
    Plot FFT frequency response with dark mode support
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for idx, (freq, mag_db, filename) in enumerate(zip(frequencies, magnitude_db, filenames)):
        # Apply smoothing if requested
        if smoothing_factor > 1:
            mag_db = signal.savgol_filter(mag_db,
                                         window_length=smoothing_factor*2+1,
                                         polyorder=3)

        color = colors[idx % len(colors)]
        ax.semilogx(freq, mag_db, label=filename, linewidth=1.5,
                   color=color, alpha=0.8)

    ax.set_xlabel('Frequency (Hz)', fontsize=10)
    ax.set_ylabel('Magnitude (dB)', fontsize=10)
    ax.set_title('FFT Analysis - Frequency Response', fontsize=12)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='best', fontsize=9)

    # Set reasonable frequency limits
    ax.set_xlim([20, 20000])

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
    ax.title.set_color('#888888')

    plt.tight_layout()
    return fig

# Main App
st.title("Guitar IR Analyzer")
st.markdown("Guitar Cabinet Impulse Response Analysis Tool")
st.markdown("Upload multiple WAV files to analyze and visualize waveforms and FFT results.")

# Sidebar for settings
st.sidebar.header("Analysis Settings")

# Display duration setting
display_duration = st.sidebar.slider(
    "Waveform Display Duration (seconds)",
    min_value=0.01,
    max_value=1.0,
    value=0.1,
    step=0.01,
    help="Duration of waveform to display in the time domain plot"
)

# FFT settings
fft_window_size = st.sidebar.selectbox(
    "FFT Window Size",
    options=[2048, 4096, 8192, 16384, 32768, 65536],
    index=3,
    help="Larger window size provides better frequency resolution"
)

# Smoothing
smoothing = st.sidebar.slider(
    "FFT Smoothing",
    min_value=1,
    max_value=50,
    value=1,
    help="Apply smoothing to FFT results (1 = no smoothing)"
)

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
                fft_window_size
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
        # Display waveforms
        st.header("Waveform (Time Domain)")
        for (time_ms, waveform), filename in zip(waveform_data, filenames):
            fig = plot_waveform(time_ms, waveform, filename)
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
