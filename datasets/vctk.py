import os
import json


class VCTK:
    def __init__(self, config):
        self.data_dir = config["path"]["corpus_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.summary_file = config["path"]["dataset_summary"]
        self.wav_exts = ['_mic2.flac', '_mic1.flac']
        self.dataset_summary = {}
        self.validate_dir()

    def validate_dir(self):
        if not os.path.isdir(self.data_dir):
            raise NotADirectoryError('{} not found!'.format(self.data_dir))
        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)
        return

    @staticmethod
    def extract_spk_fid(filename):
        """
        :param filename:
        :return: spk_name, fid
        """
        basename = filename.split('/')[-1]
        spk = basename.split('_')[0]
        fid = '{}_{}'.format(basename.split('_')[0], basename.split('_')[1])
        return spk, fid

    def get_wav_path(self, fid):
        """
        :param fid: e.g., p225_001
        :return: wav_path, textgrid_path
        """
        spk = fid.split('_')[0]
        wav_path = os.path.join(
            self.data_dir, 'wav48_silence_trimmed', spk, '{}{}'.format(fid, self.wav_exts[0]))
        if not os.path.isfile(wav_path):
            wav_path = os.path.join(
                self.data_dir, 'wav48_silence_trimmed', spk, '{}{}'.format(fid, self.wav_exts[1]))
        if not os.path.isfile(wav_path):
            wav_path = None
        return wav_path

    def write_dataset_info(self):
        with open(self.summary_file, 'w') as f:
            json.dump(self.dataset_summary, f, sort_keys=True, indent=4)
        return

    def write_summary(self):
        dataset_summary = {}
        for root, dirs, files in os.walk(self.data_dir, followlinks=True):
            for basename in files:
                if basename[-10:] in self.wav_exts:
                    filename = os.path.join(root, basename)
                    spk, fid = self.extract_spk_fid(filename)
                    wav_path = self.get_wav_path(fid)
                    if wav_path is not None:
                        if spk not in dataset_summary.keys():
                            dataset_summary[spk] = {}
                            dataset_summary[spk][fid] = {'wav': wav_path}
                        else:
                            dataset_summary[spk][fid] = {'wav': wav_path}
        self.dataset_summary = dataset_summary
        self.write_dataset_info()
        return
