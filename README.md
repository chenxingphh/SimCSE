# SimCSE
论文[SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://aclanthology.org/2021.emnlp-main.552/)的代码实现。SimCSE分为无监督和有监督2个版本：
* 无监督: 将文本输入到一个包含Dropout层的网络2次，就可以得到2个语义近似的表征 $ (x,x^{+}) $
* 有监督：对于SNLI数据集，将标签为`entailment`当作正样本，将标签为`contradiction`当作负样本，将数据扩充为三元组$(x,x^{+},x^{-})$

# 数据集

有监督数据集：`CNSD_SNLI`. 该数据集是将英文的`SNLI`翻译为中文用来扩充中文语言相似性的数据集<br>
 ```
数据示例:<br>
'sentence1': '一只长着长毛的白色狗跳跃着去抓一个红绿相间的玩具。', 'sentence2': '一只动物跳跃着去捕捉一个物体。', 'gold_label': 'entailment'<br> 'sentence1': '两位医生为病人做手术。', 'sentence2': '两名医生正在对一名男子进行手术。', 'gold_label': 'neutral'
 ```

无监督数据集：`CNSD_STS_B`. 该数据集也是将英文的`STS_B`翻译为中文用来扩充中文语言相似性的数据集<br>
```
数据示例:<br>
2012test-0001||一架飞机要起飞了。||一架飞机正在起飞。||5<br>
2012test-0004||一个男人在吹一支大笛子。||一个人在吹长笛。||3
```

# 实验结果

