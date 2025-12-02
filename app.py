import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import io

# Page configuration
st.set_page_config(
    page_title="Guitar IR Analyzer V2.0",
    page_icon="🔈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to hide default footer
st.markdown("""
<style>
    /* Hide Streamlit footer */
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

def calculate_waveform(audio_data, sample_rate, display_duration):
    """
    Calculate waveform data for display

    Args:
        audio_data: Audio samples
        sample_rate: Sample rate in Hz
        display_duration: Duration to display (seconds)

    Returns:
        tuple: (time_array, waveform)
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

    return time_array, waveform

def calculate_fft(audio_data, sample_rate, fft_duration, fft_window_size, window_function='boxcar'):
    """
    Calculate FFT data for frequency analysis

    Args:
        audio_data: Audio samples
        sample_rate: Sample rate in Hz
        fft_duration: Duration to use for FFT analysis (seconds)
        fft_window_size: Minimum FFT window size
        window_function: Window function name for FFT

    Returns:
        tuple: (frequencies, magnitude_db, fft_info)
    """
    # Ensure mono
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Normalize
    audio_data = audio_data / np.max(np.abs(audio_data))

    # Calculate target samples from FFT duration (no clipping by file length)
    target_fft_samples = int(fft_duration * sample_rate)

    # Actual data samples to read (limited by file length)
    actual_data_samples = min(target_fft_samples, len(audio_data))
    fft_data = audio_data[:actual_data_samples]

    # Determine FFT size based on TARGET duration (not clipped)
    # This allows zero-padding for short files to achieve desired frequency resolution
    # Round up to next power of 2 for efficiency
    fft_size_from_duration = 2 ** int(np.ceil(np.log2(target_fft_samples)))

    # Use the larger of duration-based size and user-specified minimum window size
    actual_fft_size = max(fft_size_from_duration, fft_window_size)

    # Apply window function for FFT
    if window_function == 'boxcar':
        window = signal.windows.boxcar(len(fft_data))
    elif window_function == 'hann':
        window = signal.windows.hann(len(fft_data))
    elif window_function == 'hamming':
        window = signal.windows.hamming(len(fft_data))
    elif window_function == 'blackman':
        window = signal.windows.blackman(len(fft_data))
    elif window_function == 'bartlett':
        window = signal.windows.bartlett(len(fft_data))
    elif window_function == 'kaiser':
        window = signal.windows.kaiser(len(fft_data), beta=8.6)
    elif window_function == 'blackmanharris':
        window = signal.windows.blackmanharris(len(fft_data))
    elif window_function == 'flattop':
        window = signal.windows.flattop(len(fft_data))
    elif window_function == 'tukey':
        window = signal.windows.tukey(len(fft_data))
    else:
        window = signal.windows.boxcar(len(fft_data))

    # Apply window to FFT data
    windowed_data = fft_data * window

    # FFT Analysis with actual FFT size
    fft_result = np.fft.rfft(windowed_data, n=actual_fft_size)
    magnitude = np.abs(fft_result)

    # Convert to dB (avoid log of zero)
    magnitude_db = 20 * np.log10(magnitude + 1e-10)

    # Frequency array based on actual FFT size
    frequencies = np.fft.rfftfreq(actual_fft_size, 1/sample_rate)

    # FFT information
    fft_info = {
        'sample_rate': sample_rate,
        'fft_size': actual_fft_size,
        'target_fft_samples': target_fft_samples,  # Target samples based on duration
        'actual_data_samples': actual_data_samples,  # Actual samples from file
        'fft_duration': fft_duration,
        'frequency_resolution': sample_rate / actual_fft_size,
        'nyquist_frequency': sample_rate / 2
    }

    return frequencies, magnitude_db, fft_info

def analyze_ir(audio_data, sample_rate, display_duration, fft_duration, fft_window_size, window_function='boxcar'):
    """
    Analyze impulse response and return waveform and FFT data
    (Legacy wrapper for backward compatibility - calls new functions)

    Args:
        audio_data: Audio samples
        sample_rate: Sample rate in Hz
        display_duration: Duration to display in waveform plot (seconds)
        fft_duration: Duration to use for FFT analysis (seconds)
        fft_window_size: FFT window size
        window_function: Window function name for FFT

    Returns:
        tuple: (time_array, waveform, frequencies, magnitude_db, fft_info)
    """
    time_array, waveform = calculate_waveform(audio_data, sample_rate, display_duration)
    frequencies, magnitude_db, fft_info = calculate_fft(audio_data, sample_rate, fft_duration, fft_window_size, window_function)
    return time_array, waveform, frequencies, magnitude_db, fft_info

def plot_waveforms(waveform_data, filenames, display_duration_ms, dpi=400, colors=None,
                   text_size=10, text_color='#333333', grid_linewidth=0.8, grid_alpha=0.3, grid_color='#808080'):
    """
    Plot all waveforms in a single graph with dark mode support

    Args:
        waveform_data: List of (time_array, waveform) tuples
        filenames: List of filenames
        display_duration_ms: Display duration in milliseconds
        dpi: Graph resolution in dots per inch
        colors: List of color tuples (R, G, B, A) in [0, 1] range
        text_size: Font size for labels and legends
        text_color: Color for text elements (hex)
        grid_linewidth: Width of grid lines
        grid_alpha: Transparency of grid lines
        grid_color: Color of grid lines (hex)
    """
    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi, constrained_layout=True)

    # Use default colors if none provided
    if colors is None:
        colors = [(0.12, 0.47, 0.71, 0.8), (1.0, 0.50, 0.05, 0.8), (0.17, 0.63, 0.17, 0.8),
                  (0.84, 0.15, 0.16, 0.8), (0.58, 0.40, 0.74, 0.8), (0.55, 0.34, 0.29, 0.8),
                  (0.89, 0.47, 0.76, 0.8), (0.50, 0.50, 0.50, 0.8)]

    for idx, ((time_ms, waveform), filename) in enumerate(zip(waveform_data, filenames)):
        color = colors[idx % len(colors)]
        ax.plot(time_ms, waveform, linewidth=1.2, color=color, label=filename)

    ax.set_xlabel('Time (ms)', fontsize=text_size, color=text_color)
    ax.set_ylabel('Amplitude (Normalized)', fontsize=text_size, color=text_color)
    ax.grid(True, alpha=grid_alpha, linewidth=grid_linewidth, color=grid_color)
    ax.legend(loc='best', fontsize=text_size-1)
    ax.tick_params(colors=text_color, labelsize=text_size-1)

    # Set x-axis range to 0 ~ display_duration_ms
    ax.set_xlim([0, display_duration_ms])

    # Set white background for light mode
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Set axis colors
    ax.spines['bottom'].set_color(text_color)
    ax.spines['top'].set_color(text_color)
    ax.spines['left'].set_color(text_color)
    ax.spines['right'].set_color(text_color)

    return fig

def generate_colors(num_files, color_scheme='default'):
    """
    Generate color palette based on selected scheme

    Args:
        num_files: Number of files (colors needed)
        color_scheme: Color scheme name

    Returns:
        List of color tuples (R, G, B, A) with values in [0, 1]
    """
    colors = []

    if color_scheme == 'default':
        # Tab10 color palette (matplotlib default)
        base_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        for i in range(num_files):
            hex_color = base_colors[i % len(base_colors)]
            # Convert hex to RGB (0-1 range)
            r = int(hex_color[1:3], 16) / 255.0
            g = int(hex_color[3:5], 16) / 255.0
            b = int(hex_color[5:7], 16) / 255.0
            colors.append((r, g, b, 0.8))

    elif color_scheme == 'alpha_blue':
        # Base color: (10, 62, 85) with alpha from 0 to 0.65
        base_r, base_g, base_b = 10 / 255.0, 62 / 255.0, 85 / 255.0
        for i in range(num_files):
            alpha = 1.0 - (1.0-0.35) * (i / max(num_files - 1, 1))
            colors.append((base_r, base_g, base_b, alpha))

    elif color_scheme == 'alpha_red':
        # Base color: (191, 12, 34) with alpha from 0 to 0.65
        base_r, base_g, base_b = 191 / 255.0, 12 / 255.0, 34 / 255.0
        for i in range(num_files):
            alpha = 1.0 - (1.0-0.35) * (i / max(num_files - 1, 1))
            colors.append((base_r, base_g, base_b, alpha))

    elif color_scheme == 'gradient':
        # Gradient from (10, 62, 85) to (255, 80, 80)
        start_r, start_g, start_b = 10 / 255.0, 62 / 255.0, 85 / 255.0
        end_r, end_g, end_b = 255 / 255.0, 80 / 255.0, 80 / 255.0

        for i in range(num_files):
            t = i / max(num_files - 1, 1)  # Linear interpolation factor
            r = start_r + (end_r - start_r) * t
            g = start_g + (end_g - start_g) * t
            b = start_b + (end_b - start_b) * t
            colors.append((r, g, b, 0.8))

    return colors

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

def plot_fft(frequencies, magnitude_db, filenames, octave_smoothing=0, dpi=400, y_min=-40, y_max=5, colors=None,
             text_size=10, text_color='#333333', grid_linewidth=0.8, grid_alpha=0.3, grid_color='#808080'):
    """
    Plot FFT frequency response with dark mode support

    Args:
        frequencies: List of frequency arrays
        magnitude_db: List of magnitude arrays in dB
        filenames: List of filenames
        octave_smoothing: Octave fraction for smoothing (0 = no smoothing)
        dpi: Graph resolution in dots per inch
        y_min: Minimum value for Y-axis (dB)
        y_max: Maximum value for Y-axis (dB)
        colors: List of color tuples (R, G, B, A) in [0, 1] range
        text_size: Font size for labels and legends
        text_color: Color for text elements (hex)
        grid_linewidth: Width of grid lines
        grid_alpha: Transparency of grid lines
        grid_color: Color of grid lines (hex)
    """
    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi, constrained_layout=True)

    # Use default colors if none provided
    if colors is None:
        colors = [(0.12, 0.47, 0.71, 0.8), (1.0, 0.50, 0.05, 0.8), (0.17, 0.63, 0.17, 0.8),
                  (0.84, 0.15, 0.16, 0.8), (0.58, 0.40, 0.74, 0.8), (0.55, 0.34, 0.29, 0.8),
                  (0.89, 0.47, 0.76, 0.8), (0.50, 0.50, 0.50, 0.8)]

    for idx, (freq, mag_db, filename) in enumerate(zip(frequencies, magnitude_db, filenames)):
        # Apply octave smoothing if requested
        if octave_smoothing > 0:
            mag_db = apply_octave_smoothing(freq, mag_db, octave_smoothing)

        # Normalize each file to 0dB max
        max_magnitude = np.max(mag_db)
        mag_db_normalized = mag_db - max_magnitude

        color = colors[idx % len(colors)]
        ax.semilogx(freq, mag_db_normalized, label=filename, linewidth=1.2, color=color)

    ax.set_xlabel('Frequency (Hz)', fontsize=text_size, color=text_color)
    ax.set_ylabel('Magnitude (dB, Normalized)', fontsize=text_size, color=text_color)
    ax.grid(True, which='both', alpha=grid_alpha, linewidth=grid_linewidth, color=grid_color)
    ax.legend(loc='best', fontsize=text_size-1)
    ax.tick_params(colors=text_color, labelsize=text_size-1)

    # Set reasonable frequency limits
    ax.set_xlim([20, 20000])

    # Set x-axis ticks to engineering format: 20, 100, 1k, 10k, 20k
    ax.set_xticks([20, 100, 1000, 10000, 20000])
    ax.set_xticklabels(['20', '100', '1k', '10k', '20k'])

    # Set Y-axis range
    ax.set_ylim([y_min, y_max])

    # Set white background for light mode
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Set axis colors
    ax.spines['bottom'].set_color(text_color)
    ax.spines['top'].set_color(text_color)
    ax.spines['left'].set_color(text_color)
    ax.spines['right'].set_color(text_color)

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

# FFT duration setting (in milliseconds)
fft_duration_ms = st.sidebar.slider(
    "FFT Analysis Duration (ms)",
    min_value=1,
    max_value=1000,
    value=100,
    step=1,
    help="Duration of audio data to use for FFT analysis (independent of waveform display)"
)
# Convert to seconds for processing
fft_duration = fft_duration_ms / 1000.0

# FFT settings
fft_window_size = st.sidebar.selectbox(
    "FFT Window Size (Minimum)",
    options=[2048, 4096, 8192, 16384, 32768, 65536],
    index=3,
    help="Minimum FFT size. Actual FFT size is determined by FFT Analysis Duration (rounded to next power of 2). Larger values ensure minimum frequency resolution."
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

# Color scheme selection
color_scheme = st.sidebar.selectbox(
    "Color Scheme",
    options=[
        "Default (Tab10)",
        "Alpha - Blue",
        "Alpha - Red",
        "Gradient Interpolation"
    ],
    index=0,
    help="Color palette for plots: Default uses Tab10, Alpha varies transparency, Gradient interpolates between two colors"
)
# Map display names to scheme names
color_scheme_map = {
    "Default (Tab10)": "default",
    "Alpha - Blue": "alpha_blue",
    "Alpha - Red": "alpha_red",
    "Gradient Interpolation": "gradient"
}
color_scheme_name = color_scheme_map[color_scheme]

# FFT Y-axis range
fft_y_min = st.sidebar.selectbox(
    "FFT Y-axis Minimum (dB)",
    options=list(range(-10, -130, -10)),  # -10, -20, -30, ..., -120
    index=3,  # Default to -40
    help="Set the minimum value for FFT plot Y-axis (maximum is fixed at +5 dB)"
)

# Export Settings Section
st.sidebar.markdown("---")
st.sidebar.header("Export Settings")

# Graph resolution (DPI) for export
graph_dpi = st.sidebar.selectbox(
    "Export Image Resolution (DPI)",
    options=[100, 200, 300, 400, 600, 800, 1000, 1200, 1500, 2000],
    index=3,  # Default to 400 DPI
    help="🔽 Resolution for downloaded images. Preview is limited by browser, but downloads will use full resolution."
)

st.sidebar.info("💡 Higher DPI = sharper prints & larger files")

# Graph Appearance Settings
st.sidebar.markdown("---")
st.sidebar.header("Graph Appearance")

# Text settings
with st.sidebar.expander("Text Settings", expanded=False):
    text_size = st.slider("Text Size", min_value=6, max_value=20, value=10, step=1,
                         help="Font size for axis labels and legends")

    st.markdown("**Text Color (RGB)**")
    text_r = st.slider("Red", min_value=0, max_value=255, value=51, key="text_r")
    text_g = st.slider("Green", min_value=0, max_value=255, value=51, key="text_g")
    text_b = st.slider("Blue", min_value=0, max_value=255, value=51, key="text_b")
    text_color = f"#{text_r:02x}{text_g:02x}{text_b:02x}"
    st.color_picker("Preview", value=text_color, disabled=True, key="text_preview")

# Grid settings
with st.sidebar.expander("Grid Settings", expanded=False):
    grid_linewidth = st.slider("Grid Line Width", min_value=0.1, max_value=3.0, value=0.8, step=0.1,
                               help="Width of grid lines")
    grid_alpha = st.slider("Grid Transparency", min_value=0.0, max_value=1.0, value=0.3, step=0.05,
                          help="Grid line transparency (0=invisible, 1=opaque)")

    st.markdown("**Grid Color (RGB)**")
    grid_r = st.slider("Red", min_value=0, max_value=255, value=128, key="grid_r")
    grid_g = st.slider("Green", min_value=0, max_value=255, value=128, key="grid_g")
    grid_b = st.slider("Blue", min_value=0, max_value=255, value=128, key="grid_b")
    grid_color = f"#{grid_r:02x}{grid_g:02x}{grid_b:02x}"
    st.color_picker("Preview", value=grid_color, disabled=True, key="grid_preview")

# File Upload Section
st.header("File Upload")

uploaded_files = st.file_uploader(
    "Drag and drop files here (Maximum 8 files)",
    type=['wav'],
    accept_multiple_files=True,
    help="Upload up to 8 WAV files - Limit 200MB per file"
)

# Initialize session state for storing analysis results and file selection
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'file_selection' not in st.session_state:
    st.session_state.file_selection = {}

# Display file list and enforce file limit
if uploaded_files:
    if len(uploaded_files) > 8:
        st.error(f"⚠️ Too many files! You uploaded {len(uploaded_files)} files. Please upload 8 files or fewer.")
        uploaded_files = uploaded_files[:8]  # Take only first 8
        st.warning("Only the first 8 files will be processed.")

    st.markdown(f"**{len(uploaded_files)} file(s) uploaded** - Select files to display:")

    # Display all files with checkboxes in a single list (no pagination)
    for i, file in enumerate(uploaded_files):
        # Initialize selection state (default: all checked)
        if file.name not in st.session_state.file_selection:
            st.session_state.file_selection[file.name] = True

        # Checkbox for each file
        is_selected = st.checkbox(
            f"{i+1}. {file.name}",
            value=st.session_state.file_selection[file.name],
            key=f"file_checkbox_{i}"
        )
        st.session_state.file_selection[file.name] = is_selected

# Add execution button
analyze_button = st.button("Start Analysis", type="primary", disabled=(not uploaded_files))

# Analysis section - only runs when button is clicked
if analyze_button and uploaded_files:
    st.success(f"Processing {len(uploaded_files)} file(s)...")

    # Store raw audio data for dynamic processing
    audio_files_data = []

    # Progress bar
    progress_bar = st.progress(0)

    for idx, uploaded_file in enumerate(uploaded_files):
        try:
            # Read WAV file
            sample_rate, audio_data = wavfile.read(io.BytesIO(uploaded_file.read()))

            # Ensure mono
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)

            # Normalize
            audio_data = audio_data / np.max(np.abs(audio_data))

            # Store raw audio data
            audio_files_data.append({
                'audio_data': audio_data,
                'sample_rate': sample_rate,
                'filename': uploaded_file.name
            })

            # Update progress
            progress_bar.progress((idx + 1) / len(uploaded_files))

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")

    # Clear progress bar
    progress_bar.empty()

    # Store raw audio data in session state for dynamic processing
    if audio_files_data:
        st.session_state.analysis_results = {
            'audio_files_data': audio_files_data
        }
        st.success("Analysis complete!")
    else:
        st.warning("No files were successfully processed.")

# Display results if available in session state
# Note: All calculations are dynamic based on current sidebar settings
if st.session_state.analysis_results is not None:
    results = st.session_state.analysis_results
    audio_files_data = results['audio_files_data']

    # Get filenames for filtering
    filenames = [file_data['filename'] for file_data in audio_files_data]

    # Filter data based on checkbox selection
    selected_indices = [i for i, name in enumerate(filenames)
                       if st.session_state.file_selection.get(name, True)]

    if not selected_indices:
        st.warning("⚠️ No files selected for display. Please check at least one file.")
    else:
        filtered_audio_files = [audio_files_data[i] for i in selected_indices]
        filtered_filenames = [filenames[i] for i in selected_indices]

        # Generate colors based on selected scheme
        num_files = len(filtered_filenames)
        plot_colors = generate_colors(num_files, color_scheme_name)

        # Dynamically calculate waveform data using current display duration
        waveform_data = []
        for file_data in filtered_audio_files:
            time_array, waveform = calculate_waveform(
                file_data['audio_data'],
                file_data['sample_rate'],
                display_duration
            )
            waveform_data.append((time_array, waveform))

        # Display waveforms in a single graph
        st.header("Waveform (Time Domain)")
        st.info(f"📊 Displaying {num_files} of {len(filenames)} file(s)")
        fig = plot_waveforms(waveform_data, filtered_filenames, display_duration_ms, graph_dpi,
                            colors=plot_colors, text_size=text_size, text_color=text_color,
                            grid_linewidth=grid_linewidth, grid_alpha=grid_alpha, grid_color=grid_color)
        st.pyplot(fig)

        # Save high-resolution image to buffer for download
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=graph_dpi, bbox_inches='tight')
        buf.seek(0)

        # Download button for high-resolution image
        st.download_button(
            label="📥 Download High-Resolution Waveform (PNG)",
            data=buf,
            file_name=f"waveform_{graph_dpi}dpi.png",
            mime="image/png",
            help=f"Download waveform plot at {graph_dpi} DPI"
        )

        plt.close(fig)

        # Display FFT analysis
        st.header("FFT Analysis (Frequency Response)")

        # Dynamically calculate FFT data using current settings
        all_frequencies = []
        all_magnitudes = []
        sample_rate = None  # Will be set from first file

        for file_data in filtered_audio_files:
            frequencies, magnitude_db, fft_info = calculate_fft(
                file_data['audio_data'],
                file_data['sample_rate'],
                fft_duration,
                fft_window_size,
                window_func_name
            )
            all_frequencies.append(frequencies)
            all_magnitudes.append(magnitude_db)

            # Get sample rate from first file for FFT info display
            if sample_rate is None:
                sample_rate = fft_info['sample_rate']
                current_fft_info = fft_info

        # Create combined FFT plot with dynamically calculated data
        fig = plot_fft(all_frequencies, all_magnitudes, filtered_filenames, smoothing, graph_dpi,
                      y_min=fft_y_min, colors=plot_colors, text_size=text_size, text_color=text_color,
                      grid_linewidth=grid_linewidth, grid_alpha=grid_alpha, grid_color=grid_color)
        st.pyplot(fig)

        # Display FFT information from dynamically calculated data
        st.markdown("### FFT Analysis Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sample Rate", f"{current_fft_info['sample_rate']:,.0f} Hz")
            st.metric("FFT Size", f"{current_fft_info['fft_size']:,}")
        with col2:
            st.metric("Target FFT Samples", f"{current_fft_info['target_fft_samples']:,}")
            st.metric("FFT Duration", f"{fft_duration_ms:.1f} ms")
        with col3:
            st.metric("Frequency Resolution", f"{current_fft_info['frequency_resolution']:.2f} Hz")
            st.metric("Nyquist Frequency", f"{current_fft_info['nyquist_frequency']:,.0f} Hz")

        # Save high-resolution image to buffer for download
        buf_fft = io.BytesIO()
        fig.savefig(buf_fft, format='png', dpi=graph_dpi, bbox_inches='tight')
        buf_fft.seek(0)

        # Download button for high-resolution image
        st.download_button(
            label="📥 Download High-Resolution FFT Plot (PNG)",
            data=buf_fft,
            file_name=f"fft_plot_{graph_dpi}dpi.png",
            mime="image/png",
            help=f"Download FFT plot at {graph_dpi} DPI"
        )

        plt.close(fig)
elif not uploaded_files:
    st.info("Please upload WAV files to begin analysis.")
