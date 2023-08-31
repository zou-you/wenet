import sys
import os
import argparse
import json
import glob
from pathlib import Path
import random
from threading import Thread

import librosa

random.seed(7)


# 获取音频时长(wav,mp3)
def get_duration(file_path):
     """
     获取mp3/wav音频文件时长
     :param file_path:
     :return:
     """
     duration = librosa.get_duration(filename=file_path)
     return duration


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="/home/zouyou/workspaces/ASR/wenet/examples/dialects/s0/metadata/粤语/朗读", help="""Input data file of dialects""")

    args = parser.parse_args()
    return args


def processing(file_path):
    source = file_path / "UTTRANSINFO.txt"
    target = file_path / "UTTRANSINFO_WITH_DURATION.txt"
    with open(source, 'r') as fr, open(target, 'w') as fw:
        for index, line in enumerate(fr.readlines()):
            if index == 0:
                fw.write(f"DURATION\t{line}")
            else:
                datas = line.strip().split('\t')
                _, sentence_id, speak_id, _, text = datas
                wav_path = file_path / "WAV" / speak_id / sentence_id
                duration = get_duration(wav_path)

                fw.write(f"[{0},{round(duration,3)}]\t{line}")



def main():
    args = get_args()
    data_dir = Path(args.data_dir)

    tasks = []
    for dir in data_dir.iterdir():
        print(dir)
        tasks.append(Thread(target=processing, args=(Path(dir), )))
        # processing(Path(dir))

    for task in tasks:
        task.start()

    for task in tasks:
        task.join()


if __name__ == '__main__':
    import time
    st = time.time()
    
    main()

    print(f"耗时：{round(time.time() - st)}s")
