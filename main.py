import json
import os

from scipy.signal import butter, sosfilt
import soundfile as sf
import librosa
import numpy as np
import math

METADATA_DIR = "metadata"
MUSIC_DIR = 'Music'
OUTPUT_DIR = 'sick mixes'

DEBUG = True


class Song:
    def __init__(self, path):
        self.data, self.sr = librosa.load(path, sr=44100, mono=False)
        metadata = json.load(open(os.path.join(METADATA_DIR, path.split('\\')[-1].replace(".mp3", ".json")),
                                  encoding='utf8'))
        self.original_bpm = metadata['general_bpm']
        self.sections = metadata['song_sections']
        self.title = metadata['title']
        self.sample_index = 0

    def read(self, x=None):
        start = self.sample_index
        if x:
            self.sample_index += x
        else:
            self.sample_index = self.data.shape[1]
        return self.data[:, start:self.sample_index]

    def read_section(self, section_index):
        start = self.seconds_to_samples(self.sections[section_index]['start'])
        self.sample_index = self.seconds_to_samples(self.sections[section_index]['end'])
        return self.data[:, start:self.sample_index]

    def reset_index(self):
        self.sample_index = 0

    def seconds_to_samples(self, seconds):
        return math.ceil(self.sr * seconds)

    def beats_to_samples(self, beats):
        return math.ceil(beats / self.original_bpm * 60 * self.sr)

    def section_duration_in_beats(self, section_index):
        seconds = self.sections[section_index]['end'] - self.sections[section_index]['start']
        return self.original_bpm / 60 * seconds


def mix2(song_a: Song, song_b: Song):
    oi = -1
    for candidate in range(1, len(song_a.sections) // 2):
        if song_a.section_duration_in_beats(-candidate) > 15:
            oi = -candidate
            break
    out_section = song_a.sections[oi]
    ii = 0
    for candidate in range(len(song_b.sections) // 2):
        if song_b.section_duration_in_beats(candidate) > 15:
            ii = candidate
            break
    in_section = song_b.sections[ii]
    if DEBUG:
        d = max(0.01, out_section['start'] - 3)
        song_a.read(song_a.seconds_to_samples(d))
        out = song_a.read(song_a.seconds_to_samples(out_section['start'] - d))
    else:
        out = song_a.read(song_a.seconds_to_samples(out_section['start']))
    fade_in_beats = min(song_a.section_duration_in_beats(oi), song_b.section_duration_in_beats(ii))
    a_fade_duration = song_a.beats_to_samples(fade_in_beats)
    b_fade_duration = song_b.beats_to_samples(fade_in_beats)
    fade_a = time_stretch_over_time(song_a.read(a_fade_duration),
                                    out_section['bpm'], out_section['bpm'], in_section['bpm'])
    song_b.read(song_b.seconds_to_samples(in_section['start']))
    fade_b = time_stretch_over_time(song_b.read(b_fade_duration),
                                    in_section['bpm'], out_section['bpm'], in_section['bpm'])
    low_ratio = out_section['section_bass'] / in_section['section_bass']
    band_ratio = out_section['section_mids'] / in_section['section_mids']
    high_ratio = out_section['section_highs'] / in_section['section_highs']
    fade_a = filter3_over_time(fade_a, song_a.sr, low_ratio, band_ratio, high_ratio, True)
    fade_b = filter3_over_time(fade_b, song_b.sr, low_ratio, band_ratio, high_ratio)
    fade_a = meld(fade_a)
    fade_b = meld(fade_b)
    if fade_a.shape[1] < fade_b.shape[1]:
        fade_b[:, :fade_a.shape[1]] += fade_a
    else:
        fade_b += fade_a[:, :fade_b.shape[1]]
    if DEBUG:
        out = np.concatenate((out, fade_b, song_b.read(song_b.seconds_to_samples(3))), axis=1)
    else:
        out = np.concatenate((out, fade_b, song_b.read()), axis=1)
    song_a.reset_index()
    song_b.reset_index()
    sf.write(f"sick mixes/{song_a.title} - {song_b.title}.wav", out.T, samplerate=song_a.sr)


def filter3_over_time(sequences, sr, low_curve, band_curve, high_curve, fade_out=False, filter_order=12):
    low_pass = butter(filter_order, 200, btype='lowpass', output='sos', fs=sr)
    band_pass = butter(filter_order, (200, 5000), btype='bandpass', output='sos', fs=sr)
    high_pass = butter(filter_order, 5000, btype='highpass', output='sos', fs=sr)
    for i in range(len(sequences)):
        sequence = sequences[i]
        lows = sosfilt(low_pass, sequence, axis=1)
        band = sosfilt(band_pass, sequence, axis=1)
        highs = sosfilt(high_pass, sequence, axis=1)
        t = i / len(sequences)
        if fade_out:
            sequences[i] = lows * (1 - t ** low_curve) + band * (1 - t ** band_curve) + highs * (1 - t ** high_curve)
        else:
            sequences[i] = lows * t ** low_curve + band * t ** band_curve + highs * t ** high_curve
    return sequences


def time_stretch_over_time(samples, original_bpm, start_bpm, end_bpm, curve=1, step_size=4096, overlap=0.2):
    sequences = []
    steps = samples.shape[1] // step_size
    bpm_base = start_bpm / original_bpm
    bpm_diff = end_bpm / start_bpm - 1
    for i in range(steps):
        sequences.append(
            librosa.effects.time_stretch(samples[:, step_size * i:step_size * (i + 1) + int(step_size * overlap)],
                                         rate=bpm_base + bpm_diff * (i / steps) ** curve)
        )
    return sequences


def meld(sequences, overlap=0.2):
    bufsize = 0
    for seq in sequences:
        bufsize += seq.shape[1]
    buf = np.zeros((2, bufsize))

    buf[:, :sequences[0].shape[1]] = sequences[0]
    prev_end = sequences[0].shape[1]
    prev_fade_start = prev_end - int(prev_end * overlap)
    for seq in sequences[1:]:
        buf[:, prev_fade_start:prev_end] *= np.linspace(1, 0, prev_end - prev_fade_start, endpoint=True)
        seq_len = seq.shape[1]
        fade_len = int(seq_len * overlap)
        newseq = np.copy(seq)
        newseq[:, :fade_len] *= np.linspace(0, 1, fade_len, endpoint=True)
        buf[:, prev_fade_start:prev_fade_start + seq_len] += newseq
        prev_end = prev_fade_start + seq_len
        prev_fade_start = prev_end - fade_len
    return buf[:, :prev_end]


if __name__ == '__main__':
    songs = [Song(os.path.join(MUSIC_DIR, fn)) for fn in os.listdir(MUSIC_DIR)]
    #songs = [Song(r"A:\Projects\PycharmProjects\DJNATOR\Music\Nublu - öölaps!.mp3"),
    #         Song(r"A:\Projects\PycharmProjects\DJNATOR\Music\Terminaator - Kuutõbine.mp3")]
    print('SONGS LOADED')
    import itertools

    for a, b in itertools.permutations(songs, 2):
        print(f'mixing {a.title} & {b.title} --- ', end='')
        mix2(a, b)
        print('DONE')
