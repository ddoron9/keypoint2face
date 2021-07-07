import librosa
import librosa.filters
import numpy as np
from scipy import signal
from scipy.io import wavfile


class Audio():
    def __init__(self, 
                preemphasize=True,  
                pre_emphasis=0.97, # filter coefficient.
                ref_level_db=20,
                signal_normalization=True,
                frame_shift_ms=None,
                min_level_db=-100,
                use_lws=False,
                allow_clipping_in_normalization=True,
                n_fft=800,
                hop_size=200,
                win_size=800,
                sample_rate=16000,
                symmetric_mels=True, 
                max_abs_value=4.,
                fmin=55,
                fmax=7600,
                num_mels=80
                ): 
 
        self.preemphasize=preemphasize
        self.pre_emphasis=pre_emphasis  # filter coefficient.
        self.ref_level_db=ref_level_db
        self.signal_normalization=signal_normalization
        self.frame_shift_ms=frame_shift_ms  # Can replace hop_size parameter. (Recommended: 12.5)
        self.min_level_db=min_level_db
        # Use LWS (https://github.com/Jonathan-LeRoux/lws) for STFT and phase reconstruction
        # It"s preferred to set True to use with https://github.com/r9y9/wavenet_vocoder
        # Does not work if n_ffit is not multiple of hop_size!!
        self.use_lws=use_lws
        self.allow_clipping_in_normalization=allow_clipping_in_normalization  # Only relevant if mel_normalization = True
        self.n_fft=n_fft  # Extra window size is filled with 0 paddings to match this parameter
        self.hop_size=hop_size  # For 16000Hz, 200 = 12.5 ms (0.0125 * sample_rate)
        self.win_size=win_size  # For 16000Hz, 800 = 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
        self.sample_rate=sample_rate  # 16000Hz (corresponding to librispeech) (sox --i <filename>)
        self.symmetric_mels=symmetric_mels
        # Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2,
        # faster and cleaner convergence)
        self.max_abs_value=max_abs_value
        # Conversions
        self._mel_basis = None
        self.fmin=fmin
        # Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To
        # test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
        self.fmax=fmax 
        self.num_mels=num_mels
        
    def load_wav(self, path, sr):
        return librosa.core.load(path, sr=sr)[0]

    def save_wav(self, wav, path, sr):
        wav *= 32767 / max(0.01, np.max(np.abs(wav)))
        # proposed by @dsmiller
        wavfile.write(path, sr, wav.astype(np.int16))

    def save_wavenet_wav(self, wav, path, sr):
        librosa.output.write_wav(path, wav, sr=sr)

    def inv_preemphasis(self, wav, k, inv_preemphasize=True):
        if inv_preemphasize:
            return signal.lfilter([1], [1, -k], wav)
        return wav
 
    def get_hop_size(self): 
        if self.hop_size is None:
            assert self.frame_shift_ms is not None
            self.hop_size = int(self.frame_shift_ms / 1000 * self.sample_rate)
        return self.hop_size

    def linearspectrogram(self, wav):
        D = self._stft(self.preemphasis(wav))
        S = self._amp_to_db(np.abs(D)) - self.ref_level_db

        if self.signal_normalization:
            return self._normalize(S)
        return S


    def preemphasis(self, wav):
        if self.preemphasize:
            return signal.lfilter([1, -self.pre_emphasis], [1], wav)
        return wav

    def _amp_to_db(self, x):
        min_level = np.exp(self.min_level_db / 20 * np.log(10))
        return 20 * np.log10(np.maximum(min_level, x))

    def melspectrogram(self, wav):
        D = self._stft(self.preemphasis(wav))
        S = self._amp_to_db(self._linear_to_mel(np.abs(D))) - self.ref_level_db

        if self.signal_normalization:
            return self._normalize(S)
        return S

    def _stft(self, y):
        if self.use_lws:
            return self._lws_processor().stft(y).T
        else:
            return librosa.stft(y=y, n_fft=self.n_fft, hop_length=self.get_hop_size(), win_length=self.win_size)

    def _lws_processor(self):
        import lws
        return lws.lws(self.n_fft, get_hop_size(), fftsize=self.win_size, mode="speech")

    def _normalize(self, S):
        if self.allow_clipping_in_normalization:
            if self.symmetric_mels:
                return np.clip((2 * self.max_abs_value) * ((S - self.min_level_db) / (-self.min_level_db)) - self.max_abs_value,
                            -self.max_abs_value, self.max_abs_value)
            else:
                return np.clip(self.max_abs_value * ((S - self.min_level_db) / (-self.min_level_db)), 0, self.max_abs_value)

        assert S.max() <= 0 and S.min() - self.min_level_db >= 0
        if self.symmetric_mels:
            return (2 * self.max_abs_value) * ((S - self.min_level_db) / (-self.min_level_db)) - self.max_abs_value
        else:
            return self.max_abs_value * ((S - self.min_level_db) / (-self.min_level_db))




    def _linear_to_mel(self, spectogram): 
        if self._mel_basis is None:
            self._mel_basis = self._build_mel_basis()
        return np.dot(self._mel_basis, spectogram)


    def _build_mel_basis(self):
        assert self.fmax <= self.sample_rate // 2
        return librosa.filters.mel(self.sample_rate, self.n_fft, n_mels=self.num_mels,
                                fmin=self.fmin, fmax=self.fmax)

     
    # Those are only correct when using lws!!! (This was messing with Wavenet quality for a long time!)
    def num_frames(self, length, fsize, fshift):
        """Compute number of time frames of spectrogram
        """
        pad = (fsize - fshift)
        if length % fshift == 0:
            M = (length + pad * 2 - fsize) // fshift + 1
        else:
            M = (length + pad * 2 - fsize) // fshift + 2
        return M


    def pad_lr(self, x, fsize, fshift):
        """Compute left and right padding
        """
        M = num_frames(len(x), fsize, fshift)
        pad = (fsize - fshift)
        T = len(x) + 2 * pad
        r = (M - 1) * fshift + fsize - T
        return pad, pad + r
 
    # Librosa correct padding
    def librosa_pad_lr(self, x, fsize, fshift):
        return 0, (x.shape[0] // fshift + 1) * fshift - x.shape[0]
  
    def _db_to_amp(self, x):
        return np.power(10.0, (x) * 0.05)
 
    def _denormalize(self, D):
        if self.allow_clipping_in_normalization:
            if self.symmetric_mels:
                return (((np.clip(D, - self.max_abs_value,
                                self.max_abs_value) +  self.max_abs_value) * - self.min_level_db / (2 * self.max_abs_value))
                        + self.min_level_db)
            else:
                return ((np.clip(D, 0, self.max_abs_value) * - self.min_level_db / self.max_abs_value) + self.min_level_db)

        if self.symmetric_mels:
            return (((D + self.max_abs_value) * - self.min_level_db / (2 * self.max_abs_value)) + self.min_level_db)
        else:
            return ((D * - self.min_level_db / self.max_abs_value) + self.min_level_db)



if __name__=="__main__":
    path = '../data/train/1135/audio.wav'
    wav = librosa.load(path, 16000)[0]
    a = Audio()
    orig_mel = a.melspectrogram(wav).T
    print(orig_mel.shape)