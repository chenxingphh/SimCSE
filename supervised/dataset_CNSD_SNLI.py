'''
@Author: Haihui Pan
@Date: 2022-4-26
@Desc: 构建SimCSE数据集(有监督)
'''
import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer, ElectraTokenizerFast
from tqdm import tqdm

'''
有监督SimCSE
1、训练集上的3元组形式 {'origin': xx, 'entailment':xx, 'contradiction':xx}
2、测试集上的2元组形式 {'sentence1':xx, 'sentence2':xx, 'gold_label':xx}
3、标签映射：{contradiction:0, neutral:1, entailment:2}
'''


class CNSD_SNLI(Dataset):
    '''
    CNSD_SNLI是通过将英文SNLI翻译过来用于扩充中文NLI任务的数据集
    '''

    def __init__(self, max_len, tokenizer, mode):
        # train, test, dev
        self.mode = mode
        # 输入句子的最大长度
        self.MAX_LEN = max_len
        # 原始数据
        self.raw_data_list = self.load_raw_data(mode)
        # 分词器
        self.tokenizer = tokenizer
        # 标签映射
        self.label_dict = {'contradiction': 0, 'neutral': 1, 'entailment': 2}

    def get_file_path(self, mode):
        '''获取数据路径'''
        if mode == 'train':
            return r'data/cnsd-snli/train_proceed.txt'
        elif mode == 'dev':
            return r'data/cnsd-snli/cnsd_snli_v1.0.dev.jsonl'
        elif mode == 'test':
            return r'data/cnsd-snli/cnsd_snli_v1.0.test.jsonl'
        else:
            raise ValueError('The value of mode can only be: train, dev or test!')

    def load_raw_data(self, mode):
        data_path = self.get_file_path(mode)
        # 保存数据结果
        data_list = []

        with open(data_path) as f:
            for i in f:
                data_list.append(json.loads(i))

        return data_list  # [:10000]

    def __len__(self):
        return len(self.raw_data_list)

    def __getitem__(self, idx):
        if self.mode == 'train':
            data = self.raw_data_list[idx]
            # 进行数字化
            data_encode = self.tokenizer([data['origin'], data['entailment'], data['contradiction']],
                                         max_length=self.MAX_LEN + 2,
                                         truncation=True,
                                         padding='max_length',
                                         return_tensors='pt')
            return data_encode
        else:
            data = self.raw_data_list[idx]
            s1 = data['sentence1']
            s2 = data['sentence2']
            score = self.label_dict[data['gold_label']]

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
