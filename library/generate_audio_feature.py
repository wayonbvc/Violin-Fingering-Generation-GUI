# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 14:16:46 2023

@author: WayonLin
"""
import numpy as np
from midiutil.MidiFile import MIDIFile
from midi2audio import FluidSynth
import librosa
from collections import defaultdict

#%%
# Convert the annotated fingering data to MIDI
# Use the song bach1 played by the first violinist in TNUA dataset as example

def generate_features(filepath, source_notes):
    
    annotated_pitch=[]
    annotated_start=[]
    annotated_duration=[]
    annotated_type=[]
    annotated_measure = []

    for note in source_notes:
        annotated_pitch.append(note["pitch"])
        annotated_start.append(float(note["time_start"]))
        annotated_duration.append(float(note["duration"]))
        annotated_type.append(int(note["polyphonic"]))

     
    output_MIDI_file = "temp_midi.mid"
    output_mp3_file = "temp_mp3.mp3"
    mf = MIDIFile(1)
    track = 0 
    tempo = 60
    onset = 0
    previous_duration = 0
    mf.addTrackName(track, onset, "Sample Track")
    mf.addTempo(track, onset, tempo)
    channel = 0
    volume = 100

    for i in range(len(annotated_pitch)):
        
        if annotated_type[i]==0:
            onset += previous_duration
            previous_duration = annotated_duration[i]
        pitch = int(annotated_pitch[i])
        duration = annotated_duration[i]
        print(pitch,onset,duration,annotated_type[i])
        mf.addNote(track, channel, pitch, onset, duration, volume)

    with open(output_MIDI_file, 'wb') as outf:
        mf.writeFile(outf)

    fs = FluidSynth()
    fs.midi_to_audio(output_MIDI_file, output_mp3_file)

    x_1, fs = librosa.load(filepath)
    x_2, fs = librosa.load(output_mp3_file)

    n_fft = 4410
    hop_size = 2205

    x_1_chroma = librosa.feature.chroma_stft(y=x_1, sr=fs, tuning=0, norm=2, hop_length=hop_size, n_fft=n_fft)
    x_2_chroma = librosa.feature.chroma_stft(y=x_2, sr=fs, tuning=0, norm=2, hop_length=hop_size, n_fft=n_fft)
    D, wp = librosa.sequence.dtw(X=x_1_chroma, Y=x_2_chroma, metric='euclidean')

    S1 = np.abs(librosa.stft(x_1,hop_length=hop_size, n_fft=n_fft))
    S2 = np.abs(librosa.stft(x_2,hop_length=hop_size, n_fft=n_fft))

    onset = 0
    previous_duration = 0
    time_list=[]
    for j in range(len(annotated_pitch)):
        if annotated_type[j]==0:
            onset += previous_duration
            previous_duration = annotated_duration[j]
        time_list.append(onset)

    midi_to_audio = defaultdict(list)
    for j in range(len(wp)):
        midi_to_audio[wp[j][1]].append(wp[j][0])

    audio_feature = []
    for j in range(len(annotated_pitch)):
        time_start = int(time_list[j]*(fs/hop_size))
        time_end = min(int((time_list[j]+annotated_duration[j])*(fs/hop_size)),x_2_chroma.shape[1]-1)
        s2_to_s1_start = midi_to_audio[time_start][-1]
        s2_to_s1_end = midi_to_audio[time_end][0]
        start = int(s2_to_s1_start)
        end = min(int(s2_to_s1_end)+1,S1.shape[1])
        audio_f = S1[:,start:end].copy()
        audio_feature.append(np.mean(audio_f,axis=1))
    audio_feature = np.array(audio_feature)

    print("get the audio")
    print(audio_feature.shape)
    return audio_feature
