import yaml
import argparse

from datasets import Preprocessor, VCTK


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    dataset = VCTK(config)
    dataset.write_summary()
    preprocessor = Preprocessor(config)
    preprocessor.build_from_path()
