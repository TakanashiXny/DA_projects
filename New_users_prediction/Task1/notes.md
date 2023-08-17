# Task1



## 任务概览

- 跑通Baseline
- 简单理解代码含义
- 补充必备的Python知识



## 基本思路

此次任务是基于训练集的样本数据，构建一个模型来预测测试集中用户的新增情况。这是一个二分类任务，其中目标是根据用户的行为、属性以及访问时间等特征，预测该用户是否属于新增用户。

为了解决一个机器学习的问题，通常需要经过以下步骤：问题分析，数据探索，数据清洗，特征工程，模型训练，模型验证，结果输出。其中，特征工程到模型训练再到模型验证是一个**闭环**。在特征工程中需要对特征进行一系列的选择和处理，将筛选出来的特征和数据加入模型训练。在验证模型的优劣程度后需要对模型作出改进。改进方式包括重新选择特征，也即重新进行特征工程，或者改变使用的模型。

此处，**机器学习可能优于深度学习**。在许多机器学习问题中，特征工程的重要性不容忽视。如果特征工程能够充分捕捉数据的**关键特征**，那么机器学习算法也能够表现很好。深度学习在某种程度上可以自动学习特征，但对于特定问题，手动设计特征可能会更有效。



在处理二分类的问题时，通常的方法有逻辑回归，决策树等。但是逻辑回归更适用于**线性**的数据，而此处的数据并没有此特征。决策树可以处理非线性数据，并且可以自动捕获特征之间的交互作用。此外，它可以生成可解释的规则，有助于理解模型如何做出决策。



## 代码解读

1. 导入所需要的库

```python
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
```

其中：

- pandas库用于数据处理与分析
- numpy库用于科学计算和多维数据操作
- DecisionTreeClassifier库用于构建决策树分类模型



2. 读取训练集和测试集

```python
train_data = pd.read_csv('用户新增预测挑战赛公开数据/train.csv')
test_data = pd.read_csv('用户新增预测挑战赛公开数据/test.csv')
```



3. 将 'udmap' 列进行**One-hot**编码

'udmap' 列的数据以字典的方式呈现。其中，键包括key1-key9，即有9个可能的特征；值为一个正整数。由于机器学习只能对**数值型变量**进行，所以用数值型变量存储字典数据显得极为重要。



此处采用One-hot编码方式进行。对于每一组数据，最多可能有9个键，所以用9维向量一定能够表示所有的情况。

例如：{"key1": 3, "key2": 2}可以表示为向量[3, 2, 0, 0, 0, 0, 0, 0, 0]



```python
def udmap_onethot(d):
    v = np.zeros(9)  # 创建一个长度为 9 的零数组
    if d == 'unknown':  # 如果 'udmap' 的值是 'unknown'
        return v  # 返回零数组
    d = eval(d)  # 将 'udmap' 的值解析为一个字典
    for i in range(1, 10):  # 遍历 'key1' 到 'key9', 注意, 这里不包括10本身
        if 'key' + str(i) in d:  # 如果当前键存在于字典中
            v[i-1] = d['key' + str(i)]  # 将字典中的值存储在对应的索引位置上
            
    return v  # 返回 One-Hot 编码后的数组
```

==eval==：去除字符串数据的引号，例如

```python
a = eval('1 + 1')    # a = 2
```

此处，`eval()`函数去除了引号，得到`1+1`，这是一个加法运算，程序将计算得到的结果赋值给`a`



```python
train_udmap_df = pd.DataFrame(np.vstack(train_data['udmap'].apply(udmap_onethot)))
test_udmap_df = pd.DataFrame(np.vstack(test_data['udmap'].apply(udmap_onethot)))
```

==apply==：对可迭代对象中每一个元素调用该函数。此处将每一个"udmap"数据变为一个9维向量。

==vstack==：将多个数组堆叠成一个多维数组，例如：

```python
a = np.vstack(
	[1, 2],
    [3, 4]
)
# 得到的结果是
[[ 1, 2 ],
 [ 3, 4 ]]
```

==DataFrame==：将数组变化为DataFrame对象



```python
train_udmap_df.columns = ['key' + str(i) for i in range(1, 10)]
test_udmap_df.columns = ['key' + str(i) for i in range(1, 10)]
```

为得到的DataFrame对象的列重新命名，从key1命名至key9



```python
train_data = pd.concat([train_data, train_udmap_df], axis=1)
test_data = pd.concat([test_data, test_udmap_df], axis=1)
```

==concat==：拼接数据。此处将得到的新的key数据直接拼接至初始数据的右侧，得到一个新的表格



4. 判断编码udmap是否为空

```python
train_data['udmap_isunknown'] = (train_data['udmap'] == 'unknown').astype(int)
test_data['udmap_isunknown'] = (test_data['udmap'] == 'unknown').astype(int)
```

此处判断标准：与字符串'unknown'比较，如果二者相等，则认为此处udmap为空

==astype==：数据类型转换。此处判断得到布尔变量，方便起见，将布尔值转化为0-1变量，也即整型变量



5. 提取eid的频次特征

```python
train_data['eid_freq'] = train_data['eid'].map(train_data['eid'].value_counts())
test_data['eid_freq'] = test_data['eid'].map(train_data['eid'].value_counts())
```

此处，eid为用户访问行为变量，需要统计每一个行为出现的频次。第一个数据的eid值为26，而26总共出现了174811次，所以26对应的频次为174811. 

==map==：映射函数，与`apply`函数类似。此处`train_data['eid'].value_counts()`得到一个Series对象，也即统计得到每个eid的个数，随后考虑每一个eid，将其对应的个数填入表格中。



6. 提取eid的标签特征

```python
train_data['eid_mean'] = train_data['eid'].map(train_data.groupby('eid')['target'].mean())
test_data['eid_mean'] = test_data['eid'].map(train_data.groupby('eid')['target'].mean())
```

此处将数据按照不同的eid进行分组，然后计算每个分组中的target平均值。

==groupby==：将数据按照某个指标进行分组



7. 提取时间戳

```python
train_data['common_ts'] = pd.to_datetime(train_data['common_ts'], unit='ms')
test_data['common_ts'] = pd.to_datetime(test_data['common_ts'], unit='ms')
```

==to_datetime==：将时间戳列转换为 datetime 类型。例如：1678932546000会被转换为2023-03-15 15:14:16。如果是13位则`unit`为毫秒, 如果是10位则为秒。



```python
train_data['common_ts_hour'] = train_data['common_ts'].dt.hour
test_data['common_ts_hour'] = test_data['common_ts'].dt.hour
```

使用`dt.hour`属性从`datetime`列中提取小时信息，并将提取的小时信息存储在新的列`'common_ts_hour'`



8. 加载决策树模型进行训练

```python
clf = DecisionTreeClassifier()
# 使用 fit 方法训练模型
# train_data.drop(['udmap', 'common_ts', 'uuid', 'target'], axis=1) 从训练数据集中移除列 'udmap', 'common_ts', 'uuid', 'target'
# 这些列可能是特征或标签，取决于数据集的设置
# train_data['target'] 是训练数据集中的标签列，它包含了每个样本的目标值
clf.fit(
    train_data.drop(['udmap', 'common_ts', 'uuid', 'target'], axis=1),  # 特征数据：移除指定的列作为特征
    train_data['target']  # 目标数据：将 'target' 列作为模型的目标进行训练
)
```

此处需要选择特征。此处`udmap`是字典数据，不能作为特征。`common_ts`和`uuid`分别为行为发生的时间与编号，均不能作为特征。`target`为目标数据，因此不能作为训练数据。

==drop==：去除指定名称的列



9. 对测试集进行预测，并保存结果到result_df中

```python
result_df = pd.DataFrame({
    'uuid': test_data['uuid'],  # 使用测试数据集中的 'uuid' 列作为 'uuid' 列的值
    'target': clf.predict(test_data.drop(['udmap', 'common_ts', 'uuid'], axis=1))  # 使用模型 clf 对测试数据集进行预测，并将预测结果存储在 'target' 列中
})
```

此处用字典的形式为DataFrame进行赋值。



10. 保存结果文件到本地

```python
result_df.to_csv('submit.csv', index=None)
```

将得到的DataFrame存入csv文件便于后续使用。`index=None`表示不存入DataFrame的索引。



## 最终结果

提交以上代码，可以得到0.62591的评分。这个评分并不理想，后续需要对模型进行优化，以期得到一个更高的评分。



## 总结

在任务一中，对于Python的部分语法有了更深的认识。也对机器学习问题的处理过程有了初步的了解。