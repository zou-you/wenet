import sys
import os
import argparse
import json
import glob
from pathlib import Path
import random
import re

random.seed(7)


DATA_TYPE = ["对话", "朗读"]


NOISE_PATTERN = re.compile(r'\[(SYSTEM|LAUGHTER|SONANT|ENS|MUSIC|\+|\*)\]')
TIME_PATTERN = re.compile(r'\[([\d.]+),([\d.]+)\]')
SYMBOL_PATTERN = re.compile(r'[.,!?;:！。，？]')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="/home/zouyou/workspaces/ASR/wenet/examples/dialects/s0/metadate/粤语", help="""Input data file of dialects""")
    parser.add_argument('--output_dir', type=str, default="/home/zouyou/workspaces/ASR/wenet/examples/dialects/s0/metadate/粤语", help="""Output dir for json file""")
    parser.add_argument('--dialect', type=str, default="粤语", help="""Input type of dialect""")

    args = parser.parse_args()
    return args


def deal_with_dialogue(root_path, audios, aid_set, audios_lst, subset, keep_symbol=False):
    for audio in audios:
        audio_path = Path(audio)
        # 设置音频名称为audio id
        audio_id = audio_path.stem
        # 判断是否有重复的音频
        if audio_id in aid_set:
            print(f"当前音频文件重复：{str(audio_path)}, \t {aid_set[audio_id]}")
            continue
        aid_set[audio_id] = str(audio_path)


        # 文本路径
        text_path = audio_path.parents[1] / "TXT" / f"{audio_id}.txt"
        if not text_path.exists():
            print(f"未找到当前音频对应的文本文件：{str(audio_path)}")
            continue
        
        # 保存文本信息（已每句话为单位）
        segments = []
        with open(text_path, 'r') as fr:
            for row, line in enumerate(fr.readlines()):
                datas = line.strip().split('\t')
                
                # 判断数据是否正确
                if len(datas) != 4:
                    print(f"{str(text_path)}文本文件的第{row + 1}行有问题")
                    continue
                be_time, speak_id, sex, text = datas

                # 去除噪声数据
                if NOISE_PATTERN.search(text):
                    continue

                # 分割时间
                match = TIME_PATTERN.match(be_time)
                if match:
                    begin_time = float(match.group(1))
                    end_time = float(match.group(2))
                else:
                    print(f"{str(text_path)}的第{row + 1}行未标注音频时段")
                    continue

                # 去除符号
                if not keep_symbol:
                    text = SYMBOL_PATTERN.sub("", text)

                segment = {
                    "sid": f"{audio_id}--{str(row).zfill(5)}",
                    "begin_time": begin_time,
                    "end_time": end_time,
                    "text": text,
                    "subset": [subset]
                }

                segments.append(segment)

        # 需要保存的音频信息
        audio_msgs = {
            "aid": audio_id,
            "path": str(audio_path.relative_to(root_path)),
            "type": "对话",
            "segments": segments
        }

        audios_lst.append(audio_msgs)


def deal_with_read(root_path, audios, texts, aid_set, audios_lst, subset):
    for audio in audios:
        audio_path = Path(audio)
        # 设置音频名称为audio id
        audio_id = audio_path.stem
        # 判断是否有重复的音频
        if audio_id in aid_set:
            print(f"当前音频文件重复：{str(audio_path)}, \t {aid_set[audio_id]}")
            continue
        aid_set[audio_id] = str(audio_path)
    
        # 判断音频对应文本是否存在
        text = texts.get(audio_id)
        if not text:
            print(f"未找到{str(audio_path)}对应的文本")
            continue
        
        # 
        segment = {
            "sid": f"{audio_id}--00000",
            "begin_time": 0,
            "end_time": -1,
            "text": text,
            "subset": [subset]
        }

        # 需要保存的音频信息
        audio_msgs = {
            "aid": audio_id,
            "path": str(audio_path.relative_to(root_path)),
            "type": "朗读",
            "segments": [segment],
        }

        audios_lst.append(audio_msgs)


def deal_with_read_text(text_path, keep_symbol=False):
    text_dct = {}
    with open(f"{text_path}", 'r') as fr:
        for row, line in enumerate(fr.readlines()):
            if row == 0:
                continue
            datas = line.strip().split('\t')
            
            # 判断数据是否正确
            if len(datas) != 5:
                print(f"当前文本文件的第{row + 1}行有问题：{str(text_path)}")
                continue
            _, sentence_id, speak_id, _, text = datas
            sentence_id = sentence_id.split('.')[0]

            # 去除噪声数据
            if NOISE_PATTERN.search(text):
                print(f"{str(text_path)}朗读文本的第{row + 1}行有噪声数据：")
                continue

            # 去除符号
            if not keep_symbol:
                text = SYMBOL_PATTERN.sub("", text)

            # sentence_id: text
            text_dct[sentence_id] = text

    return text_dct



def deal_with_all(data_path, output_path, dialect, ratio=0.8):

    audios_lst = []
    for type in DATA_TYPE:
        aid_set = dict()
        path = data_path / f"{type}"
        print(f"当前处理路径: {str(path)}")

        if type == "对话":
            # 划分数据集（train, dev）
            audios = glob.glob(str(path / '*/WAV/*.wav'))
            total = len(audios)
            offset = int(total * ratio)

            train_audios = audios[: offset]
            dev_audios = audios[offset: ]
            aid_set = dict()
            
            deal_with_dialogue(data_path, train_audios, aid_set, audios_lst, subset='train')
            deal_with_dialogue(data_path, dev_audios, aid_set, audios_lst, subset='dev')

        else:
            # 划分数据集（train, dev, test）
            train_dev_audios = []
            train_dev_texts = {}
            for dir in path.iterdir():
                # 待处理的文本路径
                text_path = dir / "UTTRANSINFO.txt"
                if not text_path.exists():
                    print(f"未找到当前音频对应的文本文件：{str(dir)}")
                    continue
                
                if str(dir).endswith('-3'):     # 处理test数据集
                    test_audios = glob.glob(str(dir / './WAV/*/*.wav'))
                    test_texts = deal_with_read_text(text_path)
                else:                            # 处理train dev数据集
                    train_dev_audios.extend(glob.glob(str(dir / './WAV/*/*.wav')))
                    temp_text = deal_with_read_text(text_path)
                    train_dev_texts.update(temp_text)

            assert len(train_dev_audios) == len(train_dev_texts), f"文本数：{len(train_dev_texts)}, 音频数：{len(train_dev_audios)}"

            # 划分数据集（train, dev）
            total = len(train_dev_audios)
            offset = int(total * ratio)
            train_audios = train_dev_audios[: offset]
            dev_audios = train_dev_audios[offset: ]

            # 处理具体文件
            deal_with_read(data_path, train_audios, train_dev_texts, aid_set, audios_lst, subset='train')
            deal_with_read(data_path, dev_audios, train_dev_texts, aid_set, audios_lst, subset='dev')
            deal_with_read(data_path, test_audios, test_texts, aid_set, audios_lst, subset='test')

    data_dct = {
        "dataset": "方言数据集",
        "dialect": dialect,
        "audios": audios_lst
    }

    # 使用 json.dump() 将数据写入 JSON 文件
    output_file = output_path / f"dialects_{dialect}.json"
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(data_dct, json_file, ensure_ascii=False, indent=4)

    print(f"{output_file} 文件保存成功")


def main():
    args = get_args()

    deal_with_all(Path(args.data_dir), Path(args.output_dir), args.dialect)



if __name__ == '__main__':
    import time
    st = time.time()
    
    main()

    print(f"耗时：{round(time.time() - st)}s")
