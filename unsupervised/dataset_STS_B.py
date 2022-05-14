'''
@Author: Haihui Pan
@Date: 2022-4-26
@Desc: 使用中文SNLI来评估SimCSE的效果(无监督)
'''
import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer, ElectraTokenizerFast
from tqdm import tqdm

'''
无监督SimCSE
1、训练集上的2元组形式 {'s1': xx, 's1':xx}
2、测试集上的3元组形式 {'s1':xx, 's2':xx, 'score':xx}
'''


class STS_B(Dataset):
    '''SimCSE测试集: 二元组形式; [a1, b1, score1, a2, b2, score2]
    '''

    def __init__(self, mode, max_len, tokenizer):
        # 数据集
        self.mode = mode
        # 输入句子的最大长度
        self.MAX_LEN = max_len
        # 分词器
        self.tokenizer = tokenizer
        # 文本数据集
        self.s1_list, self.s2_list, self.score_list = self.load_raw_data(mode)

    def get_file_path(self, mode):
        '''获取数据路径'''
        if mode == 'train':
            return r'data\cnsd-sts-b\cnsd-sts-train.txt'
        elif mode == 'dev':
            return r'data\cnsd-sts-b\cnsd-sts-dev.txt'
        elif mode == 'test':
            return r'data\cnsd-sts-b\cnsd-sts-test.txt'
        else:
            raise ValueError('The value of mode can only be: train, dev or test!')

    def load_raw_data(self, mode):
        data_path = self.get_file_path(mode)

        s1_list, s2_list, score_list = [], [], []
        with open(data_path, encoding='utf-8') as f:
            for i in f:
                _, s1, s2, score = i.strip().split("||")

                s1_list.append(s1)
                s2_list.append(s2)
                score_list.append(int(score))

        # 如果是训练集
        if self.mode == 'train':
            # 将s1_list和s2_list的数据进行合并
            s1_list.extend(s2_list)
            return s1_list, None, None
        else:
            return s1_list, s2_list, score_list

    def __len__(self):
        return len(self.s1_list)

    def __getitem__(self, idx):

        if self.mode == 'train':
            s1 = self.s1_list[idx]

            # 创建训练集时，直接复制样本2份，前馈计算的时候dropout的效果是不同的
            data_encode = self.tokenizer([s1, s1],
                                         max_length=self.MAX_LEN + 2,
                                         truncation=True,
                                         padding='max_length',
                                         return_tensors='pt')
            return data_encode
        else:
            s1, s2, score = self.s1_list[idx], self.s2_list[idx], self.score_list[idx]

            s1 = self.tokenizer(s1,
                                max_length=self.MAX_LEN + 2,
                                truncation=True,
                                padding='max_length',
                                return_tensors='pt')

            s2 = self.tokenizer(s2,
                                max_length=self.MAX_LEN + 2,
                                truncation=True,
                                padding='max_length',
                                return_tensors='pt')

            return s1, s2, score
