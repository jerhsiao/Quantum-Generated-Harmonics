!pip install python-osc
!pip install librosa
%matplotlib inline
# run the cell to import pythonosc
from pythonosc.udp_client import SimpleUDPClient

import librosa
import random
import qiskit
import qiskit_aer 
from qiskit import *
from qiskit_aer import *
from qiskit.visualization import plot_distribution, plot_histogram

import numpy as np
import matplotlib as mpl

import matplotlib.pyplot as plt

from scipy.io.wavfile import read, write
from scipy.signal import butter, lfilter, stft, istft
import os
import time


# wanted to implement some sort of lowpass filter to soften audio, so did some research online
def butter_lowpass(cutoff, fs, order = 5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype = 'low', analog=False)
    return b, a
def lowpass_filter(data, cutoff, fs, order = 5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# because sometimes modifying audio data or adding in generated frequencies can be harsh at the beginning, I added a fade-in
def apply_fade_in(audio_data, fade_duration, rate):
    fade_samples = int(fade_duration * rate)
    fade_curve = np.linspace(0, 1, fade_samples)
    audio_data[:fade_samples] *= fade_curve
    return audio_data

# function to plot audio data 
def plot_audio_data(audio_data, rate, title="Audio Data"):
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, len(audio_data) / rate, num=len(audio_data)), audio_data)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()


""" 
Originally, I wanted to use fourier transform to somehow get the 
frequencies that would match for pitch mapping, but running it on the whole file data created
very small frequencies I couldn't use.
Then, I used short-term fourier transform, but that didn't generate a nice result either.
I then decided to look online for another way to find the key of a wave file like librosa. 

def fourier_transform(audio_data, rate):
    # perform short term fourier transform to convert audio signal from time domain to frequency domain 
    transformed = np.fft.fft(audio_data)
    frequencies = np.fft.fftfreq(len(transformed), 1/ rate) # convert into hertz
    print("frequencies", frequencies)
    return transformed, frequencies

def st_fourier_transform(audio_data, rate):
    f, t, Zxx = stft(audio_data, rate)
    return f, t, Zxx
    # here, f is an array of sample frequencies, t is segment times, and Zxx is the STFT of x according to scipy signal documentation
"""

def detect_key(audio_data, rate, num_frequencies = 16):
    # this function utilizes librosa to extract the key of the file and prominent pitches used
    # this computes chroma feature of audio data, meaning "pitch class profile" for each frame in audio
    chroma = librosa.feature.chroma_cqt(y=audio_data, sr=rate)

    # calculates mean value of each chroma bin across all frames in audio to give us a single value
    chroma_mean = chroma.mean(axis=1)
    # These two lines identify index of most prominent pitch class, and then midi to note converts it to a musical key
    key_idx = chroma_mean.argmax()
    key = librosa.midi_to_note(key_idx * 12) # we multiply by 12 to adjust index to match MIDI note number

    # piptrack computes pitch and magnitude of each frame in the audio
    pitches, magnitudes = librosa.core.piptrack(y=audio_data, sr=rate)
    prominent_pitches = []
    # this for loop finds the frequency with highest magnitude and appends it to prominent_pitchs
    for t in range(magnitudes.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:  # Ignore zero frequencies
            prominent_pitches.append(pitch)
    # this part filters for high pitches and unique pitches
    prominent_pitches = np.array(prominent_pitches)
    prominent_pitches = np.unique(prominent_pitches)
    prominent_pitches = prominent_pitches[prominent_pitches < 1000]  # Remove high frequencies
    
    # Sort by prominence and select top `num_frequencies`
    if len(prominent_pitches) > num_frequencies:
        prominent_pitches = prominent_pitches[:num_frequencies]
    
    return key, prominent_pitches      
def create_pitch_mapping_from_key(key, prominent_pitches):
    tonic = librosa.note_to_hz(key)
    pitch_mapping = {}
    intervals = [1, 1.25, 1.5, 1.75, 1, 1, 2, 2.5, 4] # to create harmonizing intervals
    # note that 1 is in there 2 times so that the original prominent pitch is played the most
    harmonic_frequencies = []
    # for each pitch in prominent pitches, we multiply by a random interval to create harmonizing notes
    for pitch in prominent_pitches:
        interval_index = random.randint(0, len(intervals) - 1)
        harmonic_frequencies.append(pitch * intervals[interval_index])
    
    harmonic_frequencies = sorted(set(harmonic_frequencies))[:16] #get rid of duplicates
    final_pitch_mapping = {}
    # create final pitch mapping based on harmonic frequencies generated
    for i, freq in enumerate(harmonic_frequencies):
        binary_representation = format(i, '04b')
        final_pitch_mapping[binary_representation] = freq
    
    return final_pitch_mapping

def map_frequencies_to_quantum_circuits(audio_data, num_qubits = 4):
    modified_audio_data = np.copy(audio_data)
    num_samples = len(audio_data)
    chunk_size = 20000
    sample_rate = 44100
    # there are over 750 thousand samples so we will do it in chunks to not take as long 
    example_qc = QuantumCircuit(4)
    for start in range(0, chunk_size*10, chunk_size):
        end = min(start + chunk_size, num_samples)
        chunk = audio_data[start:end]
        chunk_len = len(chunk)
        # Apply quantum circuit modifications to a subset of the chunk
        save_counts = []
        for i in range(0, chunk_len, int(chunk_len / (sample_rate // chunk_size))):
            sample = chunk[i]
            print(sample)
            qc = QuantumCircuit(num_qubits)
            qc.h(range(num_qubits))
            qc.u(sample, sample, np.abs(sample), range(num_qubits)) # applies a U gate, taking three parameters and applying them to each qubit
            # the sample value and its absolute value are used as parameters

            # Add controlled phase shifts and frequency shifts
            qc.cp(np.pi / 2, 0, 1)  # Controlled phase shift if the control qubit is in the |1⟩ state
            qc.cx(1, 2)  # CNOT gate between qubits 1 and 2, flips state of target qubit if control qubit is in the |1⟩ state
            qc.rx(sample / 50, 2)  # RX gate for phase shift, rotates around X-axis to qubit 2
            qc.cx(2, 3)  # CNOT gate
            qc.ry(sample / 100, 1)  # RY gate for phase shift, rotation around y axis
            qc.rz(sample / 150, 3)  # RZ gate for phase shift, rotation around z
            # i experimented with adding gates to add more transformations to the sound, I wanted the original wave file to be manipulated enough
            
            qc.measure_all()
            if (start == 40000):
                print("Example Quantum Circuit at chunk 40000:")
                print(qc)
            simulator = Aer.get_backend('qasm_simulator')
            qc_transpiled = transpile(qc, simulator)
            result = simulator.run(qc_transpiled, shots=1024).result()
            counts = result.get_counts(qc)

            # each count here represents the number of times each possible state was measured
            for outcome in counts:
                phase_shift = sum(int(bit) for bit in outcome) * 2 ** 5.0  # phase shift
                freq_shift = (sum(int(bit) for bit in outcome) - num_qubits / 2) * 2 ** 5.0  # drastic frequency shift
                random_noise = np.random.normal(0.6, 0.9)  # Slight random noise
                amplitude_mod = np.random.uniform(0.6, 0.9)  # vary amplitude
                modified_sample = amplitude_mod * (sample + random_noise) * np.exp(1j * (freq_shift + phase_shift))
                chunk[i] = np.real(modified_sample)  # Ensure the result is real
        modified_audio_data[start:end] = chunk
    plot_audio_data(modified_audio_data, 44100, title="Quantum Modified Audio Data")

    return modified_audio_data


def decide_delay_application():
    # i want to apply another quantum circuit to decide whether or not to apply delay, so the output (0 or 1) from counts indicates whether or not to apply
    
    qc = QuantumCircuit(1, 1) 
    qc.h(0)  
    qc.measure(0, 0) 

    simulator = Aer.get_backend('qasm_simulator')
    result = simulator.run(transpile(qc, simulator), shots=1000).result()
    counts = result.get_counts(qc)
    
    # Get the outcome (0 or 1)
    outcome = int(list(counts.keys())[0])
    return outcome
    
def apply_delay(audio_data, delay_time, rate):
    # this line calculates number of samples that correspond to desired delay time
    delay_samples = int(rate * delay_time)
    # this creates array to hold delayed signal. 
    delayed_audio = np.zeros(len(audio_data) + delay_samples)
    for i in range(len(audio_data)):
        """ for each sample in original audio data, it adds delayed position to current sample of original audio 
         with reduced amplitude (0.5) It only applies delay if i + delay_samples is 
         within bounds of delayed_audio array
        """
        delayed_audio[i] += audio_data[i]
        if i + delay_samples < len(delayed_audio):
            delayed_audio[i + delay_samples] += 0.5 * audio_data[i]
    return np.int16(delayed_audio / np.max(np.abs(delayed_audio)) * 32767)
# i wanted to add reverb to my generated frequencies to smooth out sounds
def add_reverb(waveform, rate, decay=0.5):
    # calculate number of samples that correspond to decay time
    reverb_samples = int(rate * decay)
    # create array to hold reverb waveform
    reverb_waveform = np.zeros(len(waveform) + reverb_samples)
    # add original wave form to reverb waveform 
    reverb_waveform[:len(waveform)] += waveform
    # add delayed and decayed copy of original waveform
    reverb_waveform[reverb_samples:] += waveform * 0.5
    # return reverb waveform cropped to original length
    return reverb_waveform[:len(waveform)]

def generate_and_apply_delay(freq_arr, duration_arr, rate):
    audio_out = np.array([])
    # goes through frequencies and durations, calculates number of samples that correspond to duration
    
    for pitch, duration in zip(freq_arr, duration_arr):
        if pitch <= 0:
            continue
        duration_samples = int(rate * duration)
        # generate array of time values from 0 to specified duration
        time_arr = np.linspace(0, duration, duration_samples, endpoint=False)
        # generate sine wave based on given pitch and time values
        # the 2 * np.pi converts pitch from Hertz to radians per second to be used in sine wave
        waveform = np.sin(2 * np.pi * pitch * time_arr)

        # add harmonics to make sound richer
        harmonic_waveform = np.sin(2 * np.pi * (pitch/2) * time_arr)
        combined_waveform = waveform + harmonic_waveform
        
        # determine whether or not to apply delay
        apply_delay_decision = decide_delay_application()
        
        if apply_delay_decision:
            combined_waveform = apply_delay(combined_waveform, 0.5, rate)
         # Ensure audio_out is large enough to accommodate the new waveform

        #add reverb
        combined_waveform = add_reverb(combined_waveform, rate)
        audio_out = np.concatenate((audio_out, combined_waveform))

    # normalize audio to range of int 16
    
    audio_out = np.int16(audio_out / np.max(np.abs(audio_out)) * 32767)
    return audio_out

def decode_quantum_measurements(counts, pitch_mapping):

    duration_mapping = {
        '00': 1.0,  # Whole note
        '01': 0.5,  # Half note
        '10': 0.25, # Quarter note
        '11': 0.125 # Eighth note
    }
    
    freq_arr = []
    duration_arr = []
    for outcome, count in counts.items():
        # for each count, we get the first four characters to look up a pitch value in pitchmatching, and then look at last four for duration
        pitch = pitch_mapping.get(outcome[:4], 261.63)
        duration = duration_mapping.get(outcome[4:], 1.0)
        freq_arr.append(pitch)
        duration_arr.append(duration)
    
    return freq_arr, duration_arr

def blend_audio(audio1, audio2, blend_factor):
    min_length = min(len(audio1), len(audio2))
    blended_audio = blend_factor * audio1[:min_length] + (1 - blend_factor) * audio2[:min_length]
    return blended_audio

# extend original audio to one minute by looping 
def extend_to_one_minute(audio, rate):
    target_length = 60 * rate  # 60 seconds
    extended_audio = np.tile(audio, (target_length // len(audio) + 1))[:target_length]
    return extended_audio
    
# extend generated audio to one minute
def generate_audio_until_one_minute(rate, pitch_mapping, target_duration = 60):
    target_length = target_duration * rate
    audio_out = np.array([])
    index = 1
    while len(audio_out) < target_length:
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.h(1)
        qc.h(2)
        qc.h(3)
        
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.ccx(0, 1, 3)
        qc.measure_all()
        # I make a new quantum circuit that generates audio samples until total duration is one minute
        # these gates intoduce randomness and complexity into the generation process
        # the h gates create superposition, the CNOT gates create entanglement between qubits, 
        # the CCX gate introduces higher-order entanglement to further increase complexity
        simulator = Aer.get_backend('qasm_simulator')
        result = simulator.run(qc, backend=simulator, shots=1000).result()
        counts = result.get_counts(qc)
        freq_arr, duration_arr = decode_quantum_measurements(counts, pitch_mapping)
        
        new_audio = generate_and_apply_delay(freq_arr, duration_arr, rate)
        audio_out = np.concatenate((audio_out, new_audio))
    audio_out = audio_out[:target_length]
    audio_out = audio_out / np.max(np.abs(audio_out))
    audio_out = audio_out * 0.1 # make it more quiet
    plot_audio_data(audio_out, 44100, title="Extended Quantum Generated Audio Data")
    return audio_out[:target_length]

def process_audio(input_file, output_file):
    rate, audio_data = read_audio(input_file)
    # extract key and prominent pitches
    key, prominent_pitches = detect_key(audio_data, rate)
    
    print("prominent pitches", prominent_pitches)
    # from key and pitch, create pitch_mapping with harmonic pitches
    pitch_mapping = create_pitch_mapping_from_key(key, prominent_pitches)

    print("pitch mapping:", pitch_mapping)
    # use quantum circuits to modify original audio data
    modified_audio_data = map_frequencies_to_quantum_circuits(audio_data)
    
    # normalie
    modified_audio_data = modified_audio_data / np.max(np.abs(modified_audio_data))

    # extend modified audio data, apply low pass filter, fade in to make it sound less harsh
    modified_audio_data = extend_to_one_minute(modified_audio_data, rate)
    modified_audio_data = lowpass_filter(modified_audio_data, cutoff=2000, fs=rate, order=6)
    modified_audio_data = apply_fade_in(modified_audio_data, fade_duration = 3.0, rate = rate)
    
    # generate audio to overlay on modified audio data that is based on the pitches from original audio data
    extended_quantum_audio = generate_audio_until_one_minute(rate, pitch_mapping)

    #normalize
    extended_quantum_audio = extended_quantum_audio / np.max(np.abs(extended_quantum_audio))

    # apply low pass filter
    extended_quantum_audio = lowpass_filter(extended_quantum_audio, cutoff=2000, fs=rate, order=6)
    
    blend_factor = 0.3 # since the generated audio is a bit loud, wanted to prioritize original modified audio, 
    # this blend factor makes generated audio 0.3 times as loud and the modified audio 0.7 times as loud
    blended_audio_data = blend_audio(extended_quantum_audio, modified_audio_data, blend_factor)
    # one more fade in just in case the generated audio has any loud noises at the beginning
    blended_audio_data = apply_fade_in(blended_audio_data, fade_duration = 3.0, rate = rate)

    blended_audio_data = np.int16(blended_audio_data / np.max(np.abs(blended_audio_data)) * 32767)
    
    write(output_file, rate, blended_audio_data)
process_audio('sample_tests/another_pad.wav', 'quantum_modified_another_pad.wav')

