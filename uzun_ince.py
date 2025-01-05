import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import butter, lfilter


def sine_wave(f, t, fs):
    samples = np.arange(t * fs) / fs
    return np.sin(2 * np.pi * f * samples)


def triangle_wave(f, t, fs):
    samples = np.arange(t * fs) / fs
    return 2 * np.abs(2 * ((samples * f) % 1) - 1) - 1


def sawtooth_wave(f, t, fs):
    samples = np.arange(t * fs) / fs
    return 2 * ((samples * f) % 1) - 1


def random_smooth_wave(f, t, fs):
    from scipy.interpolate import interp1d
    num_points = int(f * t)
    random_points = np.random.uniform(-1, 1, num_points)
    time_points = np.linspace(0, t, num_points)
    interp = interp1d(time_points, random_points, kind='cubic')
    samples = np.linspace(0, t, int(fs * t))
    return interp(samples)


def combined_wave(f, t, fs):
    num_samples = int(t * fs)

    sawtooth = sawtooth_wave(f, t, fs)
    triangle = triangle_wave(f, t, fs)

    waveforms = [sawtooth, triangle]
    waveforms = [wave[:num_samples] for wave in waveforms]

    combined_wave = np.sum(waveforms, axis=0)

    combined_wave = np.clip(combined_wave, -1.0, 1.0)

    return combined_wave


def low_pass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y


def apply_envelope(waveform, fs, attack, decay, sustain, release):
    total_samples = len(waveform)
    attack_samples = int(attack * fs)
    decay_samples = int(decay * fs)
    release_samples = int(release * fs)

    sustain_samples = max(0, total_samples - (attack_samples + decay_samples + release_samples))

    envelope = np.concatenate([
        np.linspace(0, 1, attack_samples, endpoint=False),  # Attack
        np.linspace(1, sustain, decay_samples, endpoint=False),  # Decay
        np.full(sustain_samples, sustain),  # Sustain
        np.linspace(sustain, 0, release_samples)  # Release
    ])

    envelope = envelope[:total_samples]

    return waveform * envelope


def note(f, t, amplitude=0.3, wave_func=combined_wave, pause=0.0, attack=0.01, decay=0.0, sustain=0.7, release=0.1):
    global song
    waveform = wave_func(f, t, fs)
    waveform = apply_envelope(waveform, fs, attack, decay, sustain, release)
    waveform = low_pass_filter(waveform, cutoff=2000, fs=fs)
    waveform *= amplitude

    if pause > 0:
        silence = np.zeros(int(pause * fs))
        waveform = np.concatenate((waveform, silence))

    song = np.concatenate((song, waveform))


fs = 44100
song = np.array([], dtype=np.float32)
C = 16.35
Cs = 17.32
D = 18.35
Ds = 19.45
E = 20.6
F = 21.83
Fs = 23.12
G = 24.5
Gs = 25.96
A = 27.5
As = 29.14
B = 30.87

note(Ds*4, 0.25)
note(G*4, 0.25)
note(G*4, 0.5)
note(D*4, 0.5)
note(F*4, 0.25)
note(D*4, 0.5)
note(C*4, 0.25)
note(C*4, 1.0)

note(C*4, 0.5)
note(G*4, 0.25)
note(Ds*4, 0.25)
note(F*4, 0.5)
note(D*4, 0.5)
note(Ds*4, 0.25)
note(D*4, 0.5)
note(C*4, 0.25)
note(C*4, 1.0)

note(As*4, 0.25)
note(As*4, 0.25)
note(As*4, 0.35, pause=0.15)
note(As*4, 0.25)
note(As*4, 0.35)
note(A*4, 0.25)
note(G*4, 0.5)
note(As*4, 0.25)
note(A*4, 0.15, pause=0.05)
note(A*4, 0.5)

note(A*4, 0.25)
note(G*4, 0.35)
note(F*4, 0.25, pause=0.15)
note(A*4, 0.25)
note(G*4, 0.15, pause=0.05)
note(G*4, 0.35)

note(G*4, 0.25)
note(F*4, 0.35)
note(Ds*4, 0.25, pause=0.15)
note(G*4, 0.25)
note(F*4, 0.15, pause=0.05)
note(F*4, 0.35)

note(F*4, 0.25)
note(Ds*4, 0.35)
note(D*4, 0.25, pause=0.15)
note(F*4, 0.25)
note(Ds*4, 0.15, pause=0.05)
note(Ds*4, 0.35)

note(Ds*4, 0.25)
note(D*4, 0.35)
note(C*4, 0.25, pause=0.15)
note(Ds*4, 0.25)
note(D*4, 0.15, pause=0.05)
note(D*4, 0.50)
note(C*4, 1.00)

note(As*3, 0.25, pause=0.15)
note(A*3, 0.15, pause=0.15)
note(C*4, 1.0)


sd.play(song, fs)
sd.wait()
sf.write("uzun_ince.wav", np.column_stack((song, song)), fs)
