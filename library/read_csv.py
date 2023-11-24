import numpy as np


def read_csv(csv_file_path):

    string_dict = {'N/A':0, 'G':1, 'D':2, 'A':3, 'E':4}
    with open(csv_file_path) as f:
        input_data = np.genfromtxt(f, delimiter=',', names=True, dtype=[('int'), ('int'),('int'), ('float'),('float'), ('<U10'), ('<U10'), ('<U10')])
        pitches = input_data['pitch']
        starts = input_data['time_start']
        durations = input_data['duration']
        polyphonic = input_data['polyphonic']
        strings = np.array([string_dict[s] for s in input_data['string']], dtype=int)
        positions = np.array([0 if p == 'N/A' else int(p) for p in input_data['position']], dtype=int)
        fingers = np.array([0 if f == 'N/A' else int(f) for f in input_data['finger']], dtype=int)
    return {'pitches': pitches, 'starts': starts, 'durations': durations, 'strings': strings, 'positions': positions, 'fingers': fingers}

