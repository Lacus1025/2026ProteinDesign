import pandas as pd

import re

# 读取文件
df = pd.read_excel('./GFP_data.xlsx')

GFP_WT = {
    "sfGFP": {
        "sequence": "MSKGEELFTGVVPILVELDGDVNGHKFSVRGEGEGDATNGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKRHDFFKSAMPEGYVQERTISFKDDGTYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNFNSHNVYITADKQKNGIKANFKIRHNVMSKGEELFTGVVPILVELDGDVNGHKFSVRGEGEGDATNGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKRHDFFKSAMPEGYVQERTISFKDDGTYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNFNSHNVYITADKQKNGIKANFKIRHNVEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSVLSKDPNEKRDHMVLLEFVTAAGITHGMDELYKEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSVLSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
        "pdb": "2B3P",
    },
    "avGFP": {
        "sequence": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
        "pdb": "2wur",
    },
    "amacGFP": {
        "sequence": "MSKGEELFTGIVPVLIELDGDVHGHKFSVRGEGEGDADYGKLEIKFICTTGKLPVPWPTLVTTLSYGILCFARYPEHMKMNDFFKSAMPEGYIQERTIFFQDDGKYKTRGEVKFEGDTLVNRIELKGMDFKEDGNILGHKLEYNFNSHNVYIMPDKANNGLKVNFKIRHNIEGGGVQLADHYQTNVPLGDGPVLIPINHYLSCQTAISKDRNETRDHMVFLEFFSACGHTHGMDELYK",
        "pdb": "7LG4",
    },
    "cgreGFP": {
        "sequence": "MTALTEGAKLFEKEIPYITELEGDVEGMKFIIKGEGTGDATTGTIKAKYICTTGDLPVPWATILSSLSYGVFCFAKYPRHIADFFKSTQPDGYSQDRIISFDNDGQYDVKAKVTYENGTLYNRVTVKGTGFKSNGNILGMRVLYHSPPHAVYILPDRKNGGMKIEYNKAFDVMGGGHQMARHAQFNKPLGAWEEDYPLYHHLTVWTSFGKDPDDDETDHLTIVEVIKAVDLETYR",
        "pdb": "2HPW",
    },
    "ppluGFP": {
        "sequence": "MPAMKIECRITGTLNGVEFELVGGGEGTPEQGRMTNKMKSTKGALTFSPYLLSHVMGYGFYHFGTYPSGYENPFLHAINNGGYTNTRIEKYEDGGVLHVSFSYRYEAGRVIGDFKVVGTGFPEDSVIFTDKIIRSNATVEHLHPMGDNVLVGSFARTFSLRDGGYYSFVVDSHMHFKSAIHPSILQNGGPMFAFRRVEELHSNTELGIVEYQHAFKTPIAFA",
        "pdb": "2G6X",
    }
}

GFP_list = {}

# print(GFP_WT)

# for i in range(10):
#     print(df.loc[i].aaMutations)
#     print(df.loc[i].GFPtype)
#     print(df.loc[i].Brightness)

def get_mutated_sequence(original_seq, mutation_str):
    print(original_seq)
    print(mutation_str)

    if mutation_str == 'WT' or not mutation_str:
        print("WT")
        return original_seq

    # 分割多个突变
    if ':' in mutation_str:
        mutations = mutation_str.split(':')
    else:
        mutations = [mutation_str]

    # 应用所有突变
    seq_list = list(original_seq)
    for mut in mutations:
        match = re.match(r'([A-Z\*])(\d+)([A-Z\*])', mut)
        if not match:
            raise ValueError(f"无法解析突变: {mut}")

        orig, pos, target = match.groups()
        pos = int(pos)

        if orig != '*' and seq_list[pos] != orig:
            print(f"ERROR: 位置 {pos} 期望 {orig}，实际是 {seq_list[pos]}")
            assert(0)

        if pos == len(seq_list):
            # 在末尾添加新氨基酸
            seq_list.append(target)
            print(f"  在末尾添加 {target}，新长度: {len(seq_list)}")
            continue

        seq_list[pos] = target
    print(''.join(seq_list))
    return ''.join(seq_list)


def get_json_sequence(file,batch=None):
    _GFP_list = []

    if batch is None:
        batch = len(file)
        print(f"序列数:{batch}")

    for i in range(batch):
        mutation_str = file.loc[i, 'aaMutations']
        gfp_type = file.loc[i, 'GFPtype'].strip()
        brightness = file.loc[i, 'Brightness']
        if gfp_type not in GFP_WT:
            print(f"警告: 未知的GFP类型 '{gfp_type}'，跳过第{i}行")
            continue
        original_seq = GFP_WT[gfp_type]['sequence']
        sequence = get_mutated_sequence(original_seq,mutation_str)
        _GFP_list.append({
            'index':i,
            'sequence':sequence,
            'type':gfp_type,
            'brightness':brightness
        })
    return _GFP_list

if __name__ == '__main__':
    print(get_json_sequence(df,5))
