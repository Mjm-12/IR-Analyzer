# Guitar IR Analyzer

A web-based tool for analyzing guitar cabinet impulse responses (IR). Upload multiple WAV files to visualize waveforms and FFT frequency responses.

## Features

- Upload multiple WAV files (up to 200MB each)
- Configurable waveform display duration
- FFT analysis with adjustable window size
- Frequency response smoothing
- Dark mode support
- Analysis on demand with execution button

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd IR-Analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default web browser.

## How to Use

1. **Upload Files**: Drag and drop WAV files or click "Browse files"
2. **Configure Settings** (sidebar):
   - Adjust waveform display duration
   - Select FFT window size
   - Apply smoothing to FFT results
3. **Start Analysis**: Click the "Start Analysis" button
4. **View Results**:
   - Waveform plots showing time domain response
   - FFT frequency response plots (20Hz - 20kHz)

## Requirements

- Python 3.8 or higher
- streamlit
- numpy
- matplotlib
- scipy

## License

MIT License