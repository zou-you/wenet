import os
import sys


def filter_by_id(idlist, input_file, output_file, field=1):
    with open(idlist, 'r') as id_file:
        seen = set(line.split()[0] for line in id_file)

    with open(input_file, 'r') as input_f, open(output_file, 'w') as output_f:
        for line in input_f:
            fields = line.split()
            if len(fields) >= field and fields[field - 1] in seen:
                output_f.write(line)
            # else:
            #     print(f"{input_file}中的{fields[0]} 不在 {idlist}中")

def subset_data_dir(utt_list, src_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)

    filter_by_id(utt_list, os.path.join(src_dir, 'utt2dur'), os.path.join(dest_dir, 'utt2dur'))
    filter_by_id(utt_list, os.path.join(src_dir, 'text'), os.path.join(dest_dir, 'text'))
    filter_by_id(utt_list, os.path.join(src_dir, 'segments'), os.path.join(dest_dir, 'segments'))

    with open(os.path.join(dest_dir, 'segments'), 'r') as segments_f:
        utt_ids = set(line.split()[1] for line in segments_f)
    with open(os.path.join(dest_dir, 'reco'), 'w') as reco_f:
        reco_f.write('\n'.join(utt_ids))

    filter_by_id(os.path.join(dest_dir, 'reco'), os.path.join(src_dir, 'wav.scp'), os.path.join(dest_dir, 'wav.scp'))

    os.remove(os.path.join(dest_dir, 'reco'))



if __name__ == '__main__':
    utt_list = sys.argv[1]
    src_dir = sys.argv[2]
    dest_dir = sys.argv[3]
    # utt_list = "/home/zouyou/workspaces/ASR/wenet/examples/dialects/s0/data/粤语/corpus/train_utt_list"
    # src_dir = "/home/zouyou/workspaces/ASR/wenet/examples/dialects/s0/data/粤语/corpus"
    # dest_dir = "/home/zouyou/workspaces/ASR/wenet/examples/dialects/s0/data/粤语/train"

    print("utt_list路径: ", utt_list)
    print("src_dir: ", src_dir)
    print("dest_dir: ", dest_dir)

    

    subset_data_dir(utt_list, src_dir, dest_dir)
