# 介绍

存放2021年共享杯比赛期间的代码以及数据，以[MIMIC IV数据库]([MIMIC (mit.edu)](https://mimic.mit.edu/#mimic-iv-citation))为数据来源，选取MIMIC-Ⅳ数据库中符合纳排标准的ICU脓毒血症（Sepsis-3）患者，提取患者人口学和实验室信息进行作为研究的数据集。根据信息熵选取贡献最大的前19个特征纳入模型。应用贝叶斯网络、Logistic回归、随机森林、支持向量机、XGBOOST、LightGBM等方法进行分类，利用十折交叉验证评估模型效。



- SQL文件夹中保存了从MIMIC IV数据的建立，数据筛选相关的`.sql`代码
- Python文件夹存放数据预处理，建模，模型评价的`.py`代码



## MIMIC IV数据库

