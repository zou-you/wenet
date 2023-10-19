import os
from pathlib import Path
import glob
import re



def rename(dir):
    pattern = re.compile(r"G0")
    for wav in glob.glob(str(dir / './WAV/*/*.wav')):
        new_name = pattern.sub('G3', wav)
        os.rename(wav, new_name)


if __name__ == '__main__':
    dir = Path('/mnt/inspurfs/user-fs/multimedia/ASR/wenet/dialects/s0/metadata/四川话/朗读/MDT2017S020-3')

    rename(dir)