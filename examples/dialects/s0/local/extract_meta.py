# Copyright 2021  Xiaomi Corporation (Author: Yongqing Wang)
#                 Mobvoi Inc(Author: Di Wu, Binbin Zhang)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
import argparse
import json


def get_args():
    parser = argparse.ArgumentParser(description="""
      This script is used to process raw json dataset of WenetSpeech,
      where the long wav is splitinto segments and
      data of wenet format is generated.
      """)
    parser.add_argument('input_json', help="""Input json file of WenetSpeech""")
    parser.add_argument('output_dir', help="""Output dir for prepared data""")

    args = parser.parse_args()
    return args


def meta_analysis(input_json, output_dir):
    input_dir = os.path.dirname(input_json)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with open(input_json, 'r') as injson:
            json_data = json.load(injson)
    except Exception:
        sys.exit(f'Failed to load input json file: {input_json}')
    else:
        if json_data['audios'] is not None:
            with open(f'{output_dir}/text', 'w') as utt2text, \
                 open(f'{output_dir}/segments', 'w') as segments, \
                 open(f'{output_dir}/utt2dur', 'w') as utt2dur, \
                 open(f'{output_dir}/wav.scp', 'w') as wavscp, \
                 open(f'{output_dir}/utt2subsets', 'w') as utt2subsets:
                """
                text: sid text                                  Y0000000000_--5llN02F84_S00000  怎么样这些日子住得还习惯吧
                segments: sid aid start_time end_time           Y0000000000_--5llN02F84_S00000  Y0000000000_--5llN02F84 20.08   24.4
                utt2dur: sid duration                           Y0000000000_--5llN02F84_S00000  4.32 
                wav.scp: aid audio_path                         Y0000000000_--5llN02F84 /workspaces/WenetSpeech/audio/train/youtube/B00000/Y0000000000_--5llN02F84.opus
                utt2subsets: sid segment                        Y0000000000_--5llN02F84_S00000  L
                reco2dur: aid duration                          Y0000000000_--5llN02F84 2494.57
                """
                for long_audio in json_data['audios']:
                    try:
                        long_audio_path = os.path.realpath(
                            os.path.join(input_dir, long_audio['path']))
                        aid = long_audio['aid']
                        segments_lists = long_audio['segments']
                        
                        assert (os.path.exists(long_audio_path))
                    except AssertionError:
                        print(f'''Warning: {aid} something is wrong,
                                  maybe AssertionError, skipped''')
                        continue
                    except Exception:
                        print(f'''Warning: {aid} something is wrong, maybe the
                                  error path: {long_audio_path}, skipped''')
                        continue
                    else:
                        wavscp.write(f'{aid}\t{long_audio_path}\n')

                        for segment_file in segments_lists:
                            try:
                                sid = segment_file['sid']
                                start_time = segment_file['begin_time']
                                end_time = segment_file['end_time']
                                dur = end_time - start_time
                                text = segment_file['text']
                                segment_subset = segment_file["subset"]
                            except Exception as E:
                                print(f'''Warning: {segment_file} something
                                          is wrong, skipped''')
                                continue
                            else:
                                utt2text.write(f'{sid}\t{text}\n')
                                segments.write(
                                    f'{sid}\t{aid}\t{start_time}\t{end_time}\n'
                                )
                                utt2dur.write(f'{sid}\t{dur}\n')
                                # utt2subsets.write(f'{sid}\t{segment_subset}\n')
                                segment_sub_names = " ".join(segment_subset)
                                utt2subsets.write(
                                    f'{sid}\t{segment_sub_names}\n')

def main():
    args = get_args()

    meta_analysis(args.input_json, args.output_dir)


if __name__ == '__main__':
    main()
