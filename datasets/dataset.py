import os
import copy
import torch
import random
import numpy as np

from torch.utils.data import Dataset


class MelDataset(Dataset):
    def __init__(self, preprocess_config, subset='train'):
        assert subset in ['train', 'val', 'test']
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.segment_size = preprocess_config["preprocessing"]["segment_size"]
        self.n_pitch_bins = preprocess_config["preprocessing"]["n_pitch_bins"]
        self.n_ene_bins = preprocess_config["preprocessing"]["n_energy_bins"]
        self.chunk_size = preprocess_config["preprocessing"]["chunk_size"]
        assert self.segment_size % self.chunk_size == 0
        meta_path = os.path.join(
            preprocess_config["path"]["preprocessed_path"], '{}-prosody.txt'.format(subset))
        self.fids, self.mel_paths, self.lf0_paths, self.energy_paths = self.process_meta(meta_path)

    def __len__(self):
        return len(self.fids)

    def chunk_shuffle(self, x, chunk_size):
        """
        :param x: [dim, time]
        :return: shuffled version
        """
        time = x.shape[1]
        dim = x.shape[0]
        x_T = x.T
        x_reshaped = np.reshape(x_T, [time // chunk_size, -1, dim])
        np.random.shuffle(x_reshaped)
        x_shuffled = np.reshape(x_reshaped, [-1, dim]).T
        return x_shuffled

    def lf0_quantize(self, lf0, eps=1e-12):
        max_val, min_val = np.max(lf0[lf0 != 0]), np.min(lf0[lf0 != 0])
        bins = np.linspace(min_val, max_val + eps, num=self.n_pitch_bins)
        lf0_q = np.digitize(lf0, bins, right=False)
        return lf0_q

    def ene_quantize(self, ene, eps=1e-12):
        max_val, min_val = np.max(ene), np.min(ene)
        bins = np.linspace(min_val, max_val + eps, num=self.n_ene_bins + 1)
        arr_q = np.digitize(ene, bins, right=False) - 1
        return arr_q

    def __getitem__(self, idx):
        fid = self.fids[idx]
        mel_path = self.mel_paths[idx].replace('/data/', '/mnt/sepc621-old/data/')
        lf0_path = self.lf0_paths[idx].replace('/data/', '/mnt/sepc621-old/data/')
        energy_path = self.energy_paths[idx].replace('/data/', '/mnt/sepc621-old/data/')
        mel = np.load(mel_path)
        mel_ext = copy.deepcopy(mel)
        lf0 = np.load(lf0_path).reshape([-1])
        lf0_q = self.lf0_quantize(lf0)
        ene = np.load(energy_path).reshape([-1])
        ene_q = self.ene_quantize(ene)
        while mel_ext.shape[1] < self.segment_size:
            mel_ext = np.concatenate([mel_ext, mel], axis=1)
        pos1 = random.randint(0, mel_ext.shape[1] - self.segment_size)
        spk_mel = mel_ext[:, pos1:pos1 + self.segment_size]
        spk_mel = self.chunk_shuffle(spk_mel, self.chunk_size)
        min_len = min(mel.shape[1], len(lf0_q), len(ene))
        mel = mel[:, :min_len]
        lf0_q = lf0_q[:min_len]
        ene_q = ene_q[:min_len]
        return fid, mel, spk_mel, lf0_q, ene_q, mel.shape[1]

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            fids = []
            mel_paths = []
            lf0_paths = []
            energy_paths = []
            for line in f.readlines():
                fid, _, mel_path, lf0_path, energy_path, *_ = line.strip("\n").split("|")
                fids.append(fid)
                mel_paths.append(mel_path)
                lf0_paths.append(lf0_path)
                energy_paths.append(energy_path)
            return fids, mel_paths, lf0_paths, energy_paths

    @staticmethod
    def collate_fn(batch):
        """
           batch: a list of data (fid, mel, mel_ext, mel.shape[1])
        """
        fids = np.array([data[0] for data in batch])
        mels = [data[1] for data in batch]
        spk_mel = np.array([data[2] for data in batch])
        lf0s = [data[3] for data in batch]
        enes = [data[4] for data in batch]
        lens = np.array([data[5] for data in batch])
        lens_batch = torch.from_numpy(lens).to(torch.int32)
        spk_mel_batch = torch.from_numpy(spk_mel).to(torch.float32)
        mel_batch = [torch.from_numpy(mel.T) for mel in mels]
        mel_batch = torch.nn.utils.rnn.pad_sequence(mel_batch, batch_first=True)
        mel_batch = mel_batch.permute((0, 2, 1))
        lf0_batch = [torch.from_numpy(lf0).unsqueeze(1) for lf0 in lf0s]
        lf0_batch = torch.nn.utils.rnn.pad_sequence(lf0_batch, batch_first=True)
        lf0_batch = lf0_batch.squeeze(2)
        ene_batch = [torch.from_numpy(ene).unsqueeze(1) for ene in enes]
        ene_batch = torch.nn.utils.rnn.pad_sequence(ene_batch, batch_first=True)
        ene_batch = ene_batch.squeeze(2)
        return fids, mel_batch, spk_mel_batch, lf0_batch, ene_batch, lens_batch
