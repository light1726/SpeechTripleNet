import os
import json
import librosa
import numpy as np
import audio as Audio

from tqdm import tqdm


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.data_summary_f = config["path"]["dataset_summary"]
        self.data_summary = self.load_dataset_info()
        self.out_dir = config["path"]["preprocessed_path"]
        self.mel_dir = os.path.join(self.out_dir, 'mels')
        self.lf0_dir = os.path.join(self.out_dir, 'lf0s')
        self.ene_dir = os.path.join(self.out_dir, 'energy')
        self.sampling_rate = config["preprocessing"]["mel_hg"]["sampling_rate"]
        self.hop_size = config["preprocessing"]["mel_hg"]["hop_size"]
        self.win_size = config["preprocessing"]["mel_hg"]["win_size"]
        self.lower_f0 = config["preprocessing"]["lower_f0"]
        self.upper_f0 = config["preprocessing"]["upper_f0"]
        self.trim_db = config["preprocessing"]["sil_top_db"]
        self.mel_extractor = Audio.MelExtractor(
            n_fft=config["preprocessing"]["mel_hg"]["n_fft"],
            num_mels=config["preprocessing"]["mel_hg"]["num_mels"],
            hop_size=config["preprocessing"]["mel_hg"]["hop_size"],
            win_size=config["preprocessing"]["mel_hg"]["win_size"],
            sampling_rate=config["preprocessing"]["mel_hg"]["sampling_rate"],
            fmin=config["preprocessing"]["mel_hg"]["fmin"],
            fmax=config["preprocessing"]["mel_hg"]["fmax"],
            top_db=config["preprocessing"]["sil_top_db"])
        self.validate_dir()

    def validate_dir(self):
        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)
        if not os.path.isdir(self.mel_dir):
            os.makedirs(self.mel_dir)
        if not os.path.isdir(self.lf0_dir):
            os.makedirs(self.lf0_dir)
        if not os.path.isdir(self.ene_dir):
            os.makedirs(self.ene_dir)
        return

    def load_dataset_info(self):
        if not os.path.isfile(self.data_summary_f):
            raise FileNotFoundError(
                '{} not exists! Please generate it first!'.format(self.data_summary_f))
        with open(self.data_summary_f, 'r') as f:
            dataset_summary = json.load(f)
        return dataset_summary

    def build_from_path(self):
        print("Processing Data ...")
        n_frames = 0
        # Compute mel-spectrogram
        out = {}
        for split in tqdm(self.data_summary.keys()):
            out[split] = []
            for fid in self.data_summary[split]:
                info, n = self.process_utterance(split, fid)
                n_frames += n
                out[split].append(info)
        print(
            "Total time: {} hours".format(
                n_frames * self.hop_size / self.sampling_rate / 3600
            )
        )
        return out

    def process_utterance(self, spk, fid):
        wav_path = self.data_summary[spk][fid]['wav']

        # Read and trim wav files
        mel_spectrogram, wav = self.mel_extractor(wav_path)

        # Compute energy and pitch
        energy = librosa.feature.rms(y=wav, frame_length=self.win_size, hop_length=self.hop_size)
        lf0 = Audio.tools.logf0(
            wav, lower_f0=self.lower_f0, upper_f0=self.upper_f0,
            sample_rate=self.sampling_rate, frame_shift_sample=self.hop_size)

        # Save files
        mel_path = os.path.join(self.mel_dir, "{}.npy".format(fid))
        np.save(mel_path, mel_spectrogram)
        lf0_path = os.path.join(self.lf0_dir, "{}.npy".format(fid))
        np.save(lf0_path, lf0)
        ene_path = os.path.join(self.ene_dir, "{}.npy".format(fid))
        np.save(ene_path, energy)
        emo = fid.split('_')[0]

        # return "|".join([fid, emo, mel_path]), mel_spectrogram.shape[1]
        return "|".join([fid, emo, mel_path, lf0_path, ene_path]), mel_spectrogram.shape[1]
