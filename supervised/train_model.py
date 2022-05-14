'''
@Author: Haihui Pan
@Date: 2022-4-26
@Desc: 基于SimCSE来训练模型
'''
import torch
import torch.nn.functional as F
from model_electra import Electra
from dataset_CNSD_SNLI import CNSD_SNLI
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import time
import scipy.stats


def simcse_supervised_loss(y_pred, t=0.05):
    # SimCSE中有监督的输入形式为:[original, entailment, contradict]
    # [o1, e1, c1, o2, e2, c2,...]
    pair_num = 3

    assert y_pred.shape[0] % pair_num == 0, "batch_size must be a multiple of pair_num!"

    # [0,3]
    row = torch.arange(0, y_pred.shape[0], pair_num).to(device)

    # [0, 1, 2, 3, 4, 5]
    col = torch.arange(y_pred.shape[0])
    # 移除能整除3的元素（移除original的标签）
    # [1, 2, 4, 5]
    col = torch.where(col % pair_num != 0)[0].to(device)

    # tensor([0, 2]) ,这个标签是针对移除o1,o2之后的e1,e2对应的index
    y_true = torch.arange(0, len(col), pair_num - 1).to(device)

    # 计算同一个batch内，任意两个样本之间的余弦相似度
    # similarities: (6, 6)
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2).to(device)
    # 选择每一个样本需要对齐的其他样本
    similarities = torch.index_select(similarities, 0, row)
    similarities = torch.index_select(similarities, 1, col)

    # 除于温度超参数
    similarities = similarities / t

    loss = F.cross_entropy(similarities, y_true)
    return torch.mean(loss)


# 单个epoch的训练过程
def train(model, dataloader, optimizer, epoch):
    model.train()

    for idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        start_time = time.time()

        # 训练集中每个数据是以[original, entailment, contradict]进行处理
        input_ids = batch['input_ids'].view(len(batch['input_ids']) * 3, -1).to(device)
        attention_mask = batch['attention_mask'].view(len(batch['attention_mask']) * 3, -1).to(device)
        token_type_ids = batch['token_type_ids'].view(len(batch['token_type_ids']) * 3, -1).to(device)

        # 前馈计算
        pred = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # 计算Loss
        loss = simcse_supervised_loss(pred)

        # 计算梯度
        loss.backward()
        loss_value = loss.data

        # 更新参数
        optimizer.step()
        scheduler.step()

        # 计算耗时
        cost_time = time.time() - start_time

        print('| epoch:{:3d} | batch:{:5d}/{:5d} | train_loss:{:8.4f} | time: {:5.2f}s'.format(epoch, idx,
                                                                                               len(dataloader),
                                                                                               loss_value, cost_time))


# 验证集上最佳的性能
best_dev_spearman = 0
# 验证集上最佳性能对应的测试集性能
best_test_spearman = 0


def evaluate_dev(model, dev_dataloader, test_dataloader):
    model.eval()

    # --------------
    # 评估验证集
    # --------------
    dev_similarity_list, dev_score_list = [], []
    with torch.no_grad():
        for idx, (s1, s2, score) in enumerate(dev_dataloader):
            # s1前馈计算
            s1_pred = model(input_ids=s1['input_ids'].view(len(s1['input_ids']), -1).to(device),
                            token_type_ids=s1['token_type_ids'].view(len(s1['token_type_ids']), -1).to(device),
                            attention_mask=s1['attention_mask'].view(len(s1['attention_mask']), -1).to(device))

            # s2前馈计算
            s2_pred = model(input_ids=s2['input_ids'].view(len(s2['input_ids']), -1).to(device),
                            token_type_ids=s2['token_type_ids'].view(len(s2['token_type_ids']), -1).to(device),
                            attention_mask=s2['attention_mask'].view(len(s2['attention_mask']), -1).to(device))

            # 计算余弦相似度
            similarity = F.cosine_similarity(s1_pred, s2_pred)
            similarity = similarity.cpu().numpy().tolist()

            # 添加结果
            dev_similarity_list.extend(similarity)
            dev_score_list.extend(score)

    # 验证集separman相关系数
    dev_spearman = scipy.stats.spearmanr(dev_similarity_list, dev_score_list).correlation

    # -----------------
    # 评估测试集
    # -----------------
    test_similarity_list, test_score_list = [], []
    with torch.no_grad():
        for idx, (s1, s2, score) in enumerate(test_dataloader):
            # s1前馈计算
            s1_pred = model(input_ids=s1['input_ids'].view(len(s1['input_ids']), -1).to(device),
                            token_type_ids=s1['token_type_ids'].view(len(s1['token_type_ids']), -1).to(device),
                            attention_mask=s1['attention_mask'].view(len(s1['attention_mask']), -1).to(device))

            # s2前馈计算
            s2_pred = model(input_ids=s2['input_ids'].view(len(s2['input_ids']), -1).to(device),
                            token_type_ids=s2['token_type_ids'].view(len(s2['token_type_ids']), -1).to(device),
                            attention_mask=s2['attention_mask'].view(len(s2['attention_mask']), -1).to(device))

            # 计算余弦相似度
            similarity = F.cosine_similarity(s1_pred, s2_pred)
            similarity = similarity.cpu().numpy().tolist()

            # 添加结果
            test_similarity_list.extend(similarity)
            test_score_list.extend(score)

    # 测试集separman相关系数
    test_spearman = scipy.stats.spearmanr(test_similarity_list, test_score_list).correlation

    # 使用验证集上最佳性能对应的测试机
    global best_dev_spearman
    best_dev_spearman = max(best_dev_spearman, dev_spearman)

    # 验证集上的最佳性能对应的测试集性能
    if best_dev_spearman == dev_spearman:
        global best_test_spearman
        best_test_spearman = test_spearman

    print('-' * 120)
    print(
        "| epoch:{:3d}| dev_spearman: {:6.4f} | test_spearman: {:6.4f} | best_dev_spearman: {:6.4f}| best_test_spearman: {:6.4f}".format(
            epoch, dev_spearman, test_spearman, best_dev_spearman, best_test_spearman))
    print('-' * 120)


if __name__ == '__main__':
    # 超参数
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 10
    batch_size = 16
    warm_up_proportion = 0.1

    # 加载模型
    model_path = r'bert-base-chinese'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = Electra(model_path, pooling_type='first_last_avg').to(device)

    # 加载训练集
    train_dataset = CNSD_SNLI(max_len=30, tokenizer=tokenizer, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, )
    # 加载验证集
    dev_data = CNSD_SNLI(max_len=30, tokenizer=tokenizer, mode='dev', )
    dev_dataloader = DataLoader(dev_data, batch_size=batch_size, shuffle=False, )
    # 加载测试集
    test_data = CNSD_SNLI(max_len=30, tokenizer=tokenizer, mode='test', )
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, )

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00003)
    # 更新次数
    total_step = len(train_dataset) * epochs // batch_size
    # 学习率计划
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_step * warm_up_proportion,
                                                num_training_steps=total_step)

    for epoch in range(1, epochs + 1):
        print("| epoch:{:3d}| learning_rate:{:6.5f}|".format(epoch, optimizer.param_groups[0]['lr']))

        train(model=model, dataloader=train_dataloader, optimizer=optimizer, epoch=epoch)

        # 评估测试集
        evaluate_dev(model, dev_dataloader=dev_dataloader, test_dataloader=test_dataloader)
