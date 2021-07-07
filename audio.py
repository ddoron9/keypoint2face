import librosa
import librosa.filters
import numpy as np
from scipy import signal
from scipy.io import wavfile

preemphasize=True  # whether to apply filter
preemphasis=0.97  # filter coefficient.
ref_level_db=20
signal_normalization=True
frame_shift_ms=None  # Can replace hop_size parameter. (Recommended: 12.5)

min_level_db=-100
# Use LWS (https://github.com/Jonathan-LeRoux/lws) for STFT and phase reconstruction
# It"s preferred to set True to use with https://github.com/r9y9/wavenet_vocoder
# Does not work if n_ffit is not multiple of hop_size!!
use_lws=False
allow_clipping_in_normalization=True  # Only relevant if mel_normalization = True

n_fft=800  # Extra window size is filled with 0 paddings to match this parameter
hop_size=200  # For 16000Hz, 200 = 12.5 ms (0.0125 * sample_rate)
win_size=800  # For 16000Hz, 800 = 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
sample_rate=16000  # 16000Hz (corresponding to librispeech) (sox --i <filename>)
symmetric_mels=True
# Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2,
# faster and cleaner convergence)
max_abs_value=4.
# max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not
# be too big to avoid gradient explosion,
# not too small for fast convergence)
# Contribution by @begeekmyfriend

def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]


def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    # proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))


def save_wavenet_wav(wav, path, sr):
    librosa.output.write_wav(path, wav, sr=sr)


def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav
#####################

def get_hop_size():
    hop_size = hop_size
    if hop_size is None:
        assert frame_shift_ms is not None
        hop_size = int(frame_shift_ms / 1000 * sample_rate)
    return hop_size


def linearspectrogram(wav):
    D = _stft(preemphasis(wav, preemphasis, preemphasize))
    S = _amp_to_db(np.abs(D)) - ref_level_db

    if signal_normalization:
        return _normalize(S)
    return S


def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav

def _amp_to_db(x):
    min_level = np.exp(min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))

def melspectrogram(wav):
    D = _stft(preemphasis(wav, preemphasis, preemphasize))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - ref_level_db

    if signal_normalization:
        return _normalize(S)
    return S

def _stft(y):
    if use_lws:
        return _lws_processor(hp).stft(y).T
    else:
        return librosa.stft(y=y, n_fft=n_fft, hop_length=get_hop_size(), win_length=win_size)

def _lws_processor():
    import lws
    return lws.lws(n_fft, get_hop_size(), fftsize=win_size, mode="speech")

def _normalize(S):
    if allow_clipping_in_normalization:
        if symmetric_mels:
            return np.clip((2 * max_abs_value) * ((S - min_level_db) / (-min_level_db)) - max_abs_value,
                           -max_abs_value, max_abs_value)
        else:
            return np.clip(max_abs_value * ((S - min_level_db) / (-min_level_db)), 0, max_abs_value)

    assert S.max() <= 0 and S.min() - min_level_db >= 0
    if symmetric_mels:
        return (2 * max_abs_value) * ((S - min_level_db) / (-min_level_db)) - max_abs_value
    else:
        return max_abs_value * ((S - min_level_db) / (-min_level_db))


# Conversions
_mel_basis = None


def _linear_to_mel(spectogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectogram)


def _build_mel_basis():
    assert fmax <= sample_rate // 2
    return librosa.filters.mel(sample_rate, n_fft, n_mels=num_mels,
                               fmin=fmin, fmax=fmax)

 
##########################################################
# Those are only correct when using lws!!! (This was messing with Wavenet quality for a long time!)
def num_frames(length, fsize, fshift):
    """Compute number of time frames of spectrogram
    """
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x, fsize, fshift):
    """Compute left and right padding
    """
    M = num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r


##########################################################
# Librosa correct padding
def librosa_pad_lr(x, fsize, fshift):
    return 0, (x.shape[0] // fshift + 1) * fshift - x.shape[0]




def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)


def _denormalize(D):
    if allow_clipping_in_normalization:
        if symmetric_mels:
            return (((np.clip(D, - max_abs_value,
                               max_abs_value) +  max_abs_value) * - min_level_db / (2 * max_abs_value))
                    + min_level_db)
        else:
            return ((np.clip(D, 0, max_abs_value) * - min_level_db / max_abs_value) + min_level_db)

    if symmetric_mels:
        return (((D + max_abs_value) * - min_level_db / (2 * max_abs_value)) + min_level_db)
    else:
        return ((D * - min_level_db / max_abs_value) + min_level_db)
