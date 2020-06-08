from pathlib import Path
import argparse
from itertools import groupby

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.cluster import SpectralClustering

from resemblyzer import preprocess_wav, VoiceEncoder, hparams


def fingerprint_from_waveform(wav):
    return VoiceEncoder().embed_utterance(wav)

def fingerprint_from_file(filepath, segment=None, sampling_rate=16000):
    fpath = Path(filepath)
    wav = preprocess_wav(fpath)
    if segment:
        wav = wav[int(segment[0] * sampling_rate):int(segment[1]) * sampling_rate]
    return VoiceEncoder().embed_utterance(wav)

def create_dummy_speaker():
    wav_fpaths = list(Path("audio_data", "librispeech_test-other").glob("**/*.flac"))
    speaker_wavs = {speaker: list(map(preprocess_wav, wav_fpaths)) for speaker, wav_fpaths in
                    groupby(tqdm(wav_fpaths, "Preprocessing wavs", len(wav_fpaths), unit="wavs"),
                            lambda wav_fpath: wav_fpath.parent.stem)}
    spk_embeds = np.array([VoiceEncoder().embed_speaker(wavs[:len(wavs)]) for wavs in speaker_wavs.values()])
    return(np.mean(spk_embeds, 0))

def speaker_diarization(**kwargs):
    if 'wav' in kwargs:
        wav = kwargs['wav']
    elif 'filepath' in kwargs:
        wav = preprocess_wav(kwargs['filepath'])

    avg_embed, cont_embeds, wav_splits = VoiceEncoder().embed_utterance(wav, return_partials=True, rate=16)

    fig, ax = plt.subplots(figsize=(6, 6))
    wav_seconds = len(wav)/hparams.sampling_rate
    timesteps = np.arange(0, wav_seconds, wav_seconds/len(cont_embeds))

    if 'speaker_embed' in kwargs:
        # compare utterance embeddings with speaker embeddings
        similarity = cont_embeds @ kwargs['speaker_embed']
        dummy_similarity = cont_embeds @ create_dummy_speaker()
        ax.plot(timesteps, similarity, 'g')
        ax.plot(timesteps, dummy_similarity, 'k--')
    else:
        # cluster utterance embeddings using Spectral Clustering
        spectral = SpectralClustering(n_clusters=2).fit_predict(cont_embeds)
        ax.plot(timesteps, spectral)

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('filename', metavar='fn', type=str,
                        help='path to audio file to be diarized')
    parser.add_argument('speaker_filename', metavar='sp_f', type=str,
                        help='path to audio file to use for speaker diarization')
    parser.add_argument('--speaker_segment', '-s', type=int, action='append',
                        help='int segment [start end] of the audio file to use for speaker diarization')
    args = vars(parser.parse_args())

    speaker = fingerprint_from_file(filepath=args.get('speaker_filename'), segment=args.get('speaker_segment'))
    speaker_diarization(filepath=args.get('filename'), speaker_embed=speaker)
