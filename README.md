Audio Editor and Analyzer
A Python graphical application for audio processing, filtering, and parameter analysis. It uses a tkinter interface to visualize waveforms, apply frequency filters, and compute acoustic metrics.

Features
Audio Playback: Load and play WAV audio files.

Real-Time Visualization: Displays the audio waveform and frequency spectrum.

Filtering: Apply Low-Pass and High-Pass filters using adjustable sliders.

Telephone Effect: Simulates a telephone frequency band (300Hz to 3400Hz).

Signal Analysis: Computes and graphs Short-Time Energy, Zero-Crossing Rate (ZCR), and Autocorrelation.

Non-Destructive Editing: Instantly restore the original audio file at any point.
Requirements
Python 3.x is required. The application depends on the following libraries:

numpy

scipy

matplotlib

sounddevice
Usage
Start the graphical interface by running the main script from your terminal:

python main.py

The application is divided into two main tabs:

Editor Audio: Use this section to load a WAV file, control playback, and apply frequency filters.

Analiza Parametrii: Use this section to generate and view mathematical analysis graphs for the currently loaded audio signal.
