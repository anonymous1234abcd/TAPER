"""
TAPER Dataset Preprocessing Script

This script processes raw accelerometer data from IMU sensors and tap pad force measurements
to extract tap events on different surface types (soft, medium, hard, harder) with configurable
downsampling rates. The processed data is saved as CSV files for machine learning tasks.

The script handles:
- Reading IMU accelerometer data and tap pad force data from XIMU3 CSV files
- Detecting tap events based on force thresholds and temporal constraints
- Extracting accelerometer magnitude windows around detected tap events
- Downsampling the data to various sampling rates (32kHz to 0.01kHz)
- Saving processed tap sequences labeled by surface type

This preprocessing is part of the TAPER (Tactile Perception and Exploration Research) dataset
for studying haptic perception and surface classification using inertial measurement units.
"""

from operator import index
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import ximu3csv

import csv

from sklearn.metrics import accuracy_score

@dataclass(frozen=True)
class Tap:
    """
    Data structure representing a single tap event.

    Attributes:
        surface (str): The surface type on which the tap occurred ('soft', 'medium', 'hard', 'harder')
        seconds (np.ndarray): Time stamps for the accelerometer samples in seconds
        timeseries (np.ndarray): Accelerometer magnitude values for the tap window
    """
    surface: str
    seconds: np.ndarray
    timeseries: np.ndarray

def csv_to_taps(surface: str, downsample_factor: int = 1) -> List[Tap]:
    """
    Convert raw CSV data to a list of Tap objects for a given surface type.

    This function orchestrates the data reading and tap detection process.

    Args:
        surface (str): Surface type ('soft', 'medium', 'hard', 'harder')
        downsample_factor (int): Factor by which to downsample the data (1 = no downsampling)

    Returns:
        List[Tap]: List of detected tap events with their accelerometer data
    """
    seconds, accelerometer, event_timestamps_sec = read_data(surface)
    return get_taps(surface, seconds, accelerometer, event_timestamps_sec, downsample_factor)

def read_data(surface: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read and process raw IMU and tap pad data for a given surface type.

    This function:
    1. Loads IMU accelerometer data from XIMU3 CSV files
    2. Processes tap pad force measurements to detect tap events
    3. Applies filtering logic to avoid false positives from closing peaks

    Args:
        surface (str): Surface type to process

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - seconds: IMU timestamp array in seconds
            - accelerometer: 3D accelerometer data (x, y, z)
            - event_timestamps_sec: Detected tap event timestamps in seconds
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TAPER_FULL_DATASET", "raw", f"{participant}", mounting_tech, surface)
    print(f"Reading data from: {path}")

    # Read all XIMU3 devices from the directory
    devices = ximu3csv.read(path)
    devices = {d.device_name: d for d in devices}

    print("Devices found:", devices.keys())

    # Select the IMU device for accelerometer data
    device = devices["Twintig CH3 IMU4"]
    print(f"Using device: {device.device_name}")

    # Initialize list to store tap pad timestamp-force pairs
    tap_pad_event_force_pairs = []

    # Determine which tap pad channel to use based on surface type
    # Different surfaces were tested with different tap pad configurations
    if surface == "soft":
        pad_to_use = 1
    elif surface == "medium":
        pad_to_use = 2
    elif surface == "hard":
        pad_to_use = 3
    elif surface == "harder":
        if participant == 6 and mounting_tech == "rings":  # Due to different configuration for this participant and mounting technique
            pad_to_use = 3
        else:    
            pad_to_use = 4
    
    # Extract tap pad data
    tap_pads_device = devices["Twintig Tap Pads"]

    # Parse tap pad CSV data: each row contains [timestamp, force_ch1, force_ch2, force_ch3, force_ch4]
    for serials in tap_pads_device.serial_accessory._csv:
        tap_pad_event_force_pairs.append((int(serials[0]), float(serials[pad_to_use])))

    # Detect tap events based on force threshold and temporal constraints
    taps_based_events = []
    previous_timestamp = tap_pad_event_force_pairs[0][0] / 1e6
    previous_force = tap_pad_event_force_pairs[0][1]
    closing_peak_timestamp = None

    # Iterate through force measurements to detect tap events
    for timestamp, force in tap_pad_event_force_pairs:
        timestamp_sec = timestamp / 1e6  # Convert microseconds to seconds

        # Detect tap event: force drops below threshold after being above threshold
        # Also ensure minimum time gap between taps and limit total number of taps
        if (force < 0.7 and len(taps_based_events) < taps_no_each and
            abs(timestamp_sec - previous_timestamp) > 0.1):  # 100ms minimum gap
            if previous_force >= 0.7:
                taps_based_events.append(timestamp_sec)
                previous_timestamp = timestamp_sec
                closing_peak_timestamp = None

        # Track closing peaks (force returning to baseline after tap)
        if force >= 0.7:
            closing_peak_timestamp = timestamp_sec

            # Optional filtering: remove taps that are immediately followed by closing peaks
            # This handles cases where users' natural tapping behavior creates double peaks
            # Uncomment the following block if needed based on visual inspection of data
            # if abs(closing_peak_timestamp - previous_timestamp) <= 0.05:  # 50ms window
            #     if taps_based_events:
            #         taps_based_events.pop()
            #         print(f"Removed tap-based event {len(taps_based_events)} at {previous_timestamp} due to closing peak at {closing_peak_timestamp}")
            #     previous_timestamp = taps_based_events[-1] if taps_based_events else 0

        previous_force = force

    # Extract IMU accelerometer data
    seconds = device.inertial.timestamp / 1e6  # Convert timestamps to seconds
    accelerometer = device.inertial.accelerometer.xyz  # 3D accelerometer data

    # Store tap pad data globally for visualization (if needed)
    global pads_data
    pads_data = tap_pad_event_force_pairs

    return seconds, accelerometer, taps_based_events

def get_taps(surface: str, seconds, accelerometer: np.ndarray, event_timestamps_sec: np.ndarray, downsample_factor: int = 1) -> List[Tap]:
    magnitude = np.linalg.norm(accelerometer, axis=1)
    
    # Downsample by taking every nth sample
    seconds = seconds[::downsample_factor]
    magnitude = magnitude[::downsample_factor]
    event_timestamps_sec = event_timestamps_sec 

    return [
        Tap(
            surface,
            seconds[s:e],
            magnitude[s:e],
        )
        for s, e in zip(*detect_impacts(surface, seconds, magnitude, event_timestamps_sec, downsample_factor))
    ]



def detect_impacts(surface: str, seconds: np.ndarray, magnitude: np.ndarray, event_timestamps_sec: np.ndarray, downsample_factor: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Define time windows around tap events for accelerometer data extraction.

    This function calculates the start and end indices for extracting accelerometer
    windows around each detected tap event. The window size adapts to the sampling
    rate and includes time before (attack) and after (decay) the tap event.

    Args:
        surface (str): Surface type (used for conditional plotting)
        seconds (np.ndarray): Downsampled timestamp array
        magnitude (np.ndarray): Downsampled accelerometer magnitude array
        event_timestamps_sec (np.ndarray): Tap event timestamps in seconds
        downsample_factor (int): Current downsampling factor

    Returns:
        Tuple[np.ndarray, np.ndarray]: Start and end indices for tap windows
    """
    # For each event timestamp, find the closest timestamp in the seconds array
    closest_indices = np.argmin(np.abs(seconds[:, np.newaxis] - event_timestamps_sec), axis=0)
    print(len(closest_indices))

    # Calculate effective sample rate after downsampling
    sample_rate = 1 / np.median(np.diff(seconds))
    print("Sample rate:", sample_rate)
    
    # Define window parameters based on sample rate
    HOLDOFF = int(sample_rate / 8)  # 125 ms - minimum time between events
    
    # Adjust window sizes based on downsampling factor for optimal capture
    if downsample_factor == 3200:
        ATTACK = int(sample_rate / 10)  # 100 ms before tap
        DECAY = int(sample_rate / 5)   # 200 ms after tap
    else:    
        ATTACK = int(sample_rate / 20)  # 50 ms before tap
        DECAY = int(sample_rate / 10)   # 100 ms after tap

    # Calculate window boundaries around each tap event
    start_indices = closest_indices - ATTACK
    end_indices = closest_indices + DECAY

    # Optional visualization for soft surface at full sampling rate
    if surface == "soft" and downsample_factor == 1:
        if True:
            plt.plot(seconds, magnitude)

            # Plot tap pad force data for comparison
            tap_seconds = [timestamp / 1e6 for (timestamp, force) in pads_data]
            tap_forces = [force for (timestamp, force) in pads_data]

            plt.plot(tap_seconds, tap_forces, label='Tap Events')
            
            # Highlight extracted windows
            for index, (impact_start, impact_end) in enumerate(zip(start_indices, end_indices)):
                plt.fill_between([seconds[impact_start], seconds[impact_end]], 
                               np.min(magnitude), np.max(magnitude), 
                               color="tab:green", alpha=0.2)

                # Label each window with its index
                center_time = (seconds[impact_start] + seconds[impact_end]) / 2
                y_pos = np.min(magnitude)
                plt.text(center_time, y_pos, str(index), 
                        horizontalalignment='center', verticalalignment='bottom', 
                        fontsize=8, fontweight='bold')
            plt.show()
                        

    return (start_indices, end_indices)



# Configuration parameters
taps_no_each = 72  # Maximum number of taps to extract per surface per participant

mounting_tech = "rings"  # Mounting technique: "tape_thick" or "rings"

surfaces = ["soft", "medium", "hard", "harder"]

# Main processing loop: iterate over participants and downsampling factors
for person in range(1, 12):  # Process participants 1 through 11
    participant = person  # Participant ID

    # Test various downsampling factors to create datasets at different sampling rates
    for factor in [1, 2, 4, 8, 16, 32, 64, 128, 256, 320, 640, 1280, 2560, 3200]:
        downsample_factor = factor  # 1 = 32kHz, 2 = 16kHz, 4 = 8kHz, etc.

        print(f"Processing data for participant: {participant} with mounting technique: {mounting_tech} (downsample_factor: {downsample_factor})")

        pads_data = []  # Reset global tap pad data storage

        # Process each surface type
        soft_taps = csv_to_taps(surfaces[0], downsample_factor)
        medium_taps = csv_to_taps(surfaces[1], downsample_factor)
        hard_taps = csv_to_taps(surfaces[2], downsample_factor)
        harder_taps = csv_to_taps(surfaces[3], downsample_factor)


        # modify as appropriate - currently it writes a csv file for each participant and mounting technique, with the surface type as the first column and the timeseries data in the following columns. you can modify this to write all surfaces in one file or to write in a different format as needed.
        import csv
        with open(f'your-processed-taps/{participant}/{mounting_tech}_data_{32/downsample_factor}khz.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                for tap in soft_taps:
                    writer.writerow([0] + tap.timeseries.tolist())
                for tap in medium_taps:
                    writer.writerow([1] + tap.timeseries.tolist())
                for tap in hard_taps:
                    writer.writerow([2] + tap.timeseries.tolist())
                for tap in harder_taps:
                    writer.writerow([3] + tap.timeseries.tolist())


        print(f"Data written to {participant}_data/{participant}_{mounting_tech}_data_{32/downsample_factor}khz.csv")