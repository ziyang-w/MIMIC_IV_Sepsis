# 介绍

存放2021年共享杯比赛期间的代码以及数据，以[MIMIC IV数据库]([MIMIC (mit.edu)](https://mimic.mit.edu/#mimic-iv-citation))为数据来源，采用贝叶斯网络预测ICU脓毒血症（Sepsis-3）患者的死亡风险。

代码包含：

1. 构建MIMIC-IV数据库以及概念的SQL代码
2. 数据预处理，数值化，分箱，缺失值处理
3. LR，RF，SVM，XGboost，LightGBM，Bayesian Network的构建以及模型评估代码
4. 模型评价，k折叫查验的绘图，以多模型绘图
5. SHAP模型可解释性



- SQL文件夹中保存了从MIMIC IV数据的建立，数据筛选相关的`.sql`代码
- Python文件夹存放数据预处理，建模，模型评价的`.py`代码



## MIMIC IV数据库

MIMIC-IV数据库是一个公开的医院系统数据库，具体的介绍可以参见[官网]([MIMIC (mit.edu)](https://mimic.mit.edu/#mimic-iv-citation))，在取得官方的授权访问之后，可以下载并构建数据库。

### 创建MIMIC数据库

1. 下载postgres数据库（当然也可以使用MySQL，但是因为MySQL的创建较慢，同时再进行后续分析时，官方没有给出MySQL的代码，需要根据postgres代码更改。）
2. 配置postgres环境变量
3. 将所有文件夹解压放入`YOUR_MIMIC_IV_FILE_PATH`路径下，路径可自由替换（Windows运行解压文件会有异常，在Linux环境下可以按照官网进行操作）
4. 在包含`create.sql` , `load.sql`的目录下运行下面的代码

```shell
cd /dData\MIMIC_IV

postgres -U postgres -f create.sql

postgres -U postgres -v mimic_data_dir= YOUR_MIMIC_IV_FILE_PATH -f load.sql
```

> tips：在运行期间可能会卡住，同时可能会遇到编码不正确的导致数据导入失败，建议将prostgres客户端编码也改成UTF-8，笔者已经在官方的`load.sql`文件中已经添加了以utf-8格式解码csv文件的代码

整体运行时间大于需要1h



[MIMIC-IV官方仓库/buildmimic/postgres](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/buildmimic/postgres)



### 构建概念

为了让MIMIC-IV数据库更加易用，原作者将常用的医学概念以及病例的提取以SQL代码共享在官方仓库中。具体参考见本节末尾。

整个代码是通过社区开发人员编写的shell脚本，从Google BigQuery代码中自动生成的符合postgres语法的SQL文件。

1. 通过GitHub下载官方代码，见本节末尾/concepts/postgres超链接
2. 将`run.sql`需要在官方sql代码目录下
3. 采用`postgres -U postgres -f run.sql`形式运行。

官方的代码会将所有的概念表保存在mimic_dervide结构下。

整体运行时间大约需要30min。

[MIMIC-IV官方仓库/concepts/postgres](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/concepts/postgres)



## 建模过程代码说明

整个`processing.py`会将缺失值处理的结果添加`_del_missing.csv`后缀，同时数据处理的结果返回到数据集所在目录的log文件夹下，按照运行的数据集的名字建立子文件夹，再以运行日期建立子文件夹，所有输出文件按照运行时间的HH命名为前缀。

`file_path/log/data_set/12-01/20_result_RF.csv`

`file_path`是数据集所在的目录，上述路径的含义是再`12月1号20点`处理的`data_set`文件中采用RF模型性能指标结果。

### 数据处理

1. 删除缺失特征，将结果重新保存到log文件夹
2. 删除缺失值（或者众数填补）
3. 变量数值化也会将数值化的编码结果以csv文件形式保存在log文件夹中
4. 采用互信息法MIC筛选模型
5. 采用K折交叉验证验证模型，并将APRF1AUC指标以csv文件保存，ROC曲线以pdf形式存储在log文件夹中。



> 在进行探索性分析时可以采用批处理代码（如Windows的power shell）循环运行，将不同的数据集和targetY作为参数传递给processing.py，可以提高数据探索的效率具体的操作可以见run.ps1。
>
> 注意：通过powershell运行python解释器需要配置环境，同时需要更改powershell运行策略



### 建模

1. 将模型性能评价指标通过调用PRF1函数返回Accuracy，Precision，Recall，F1，AUC指标
2. k折交叉验证曲线可以通过`processing.py -> plot_ROC_kfold()`函数绘制并保存
3. 多模型结果可以通过`plot_kmodel.py -> plot_ROC_kmodel()`函数绘制并保存

> 在进行网格搜索时，可以改写`find_cut_num.py`文件实现，该文件原本用于搜索研究中最适合的分箱数量和纳入模型的变量，其将模型的结果存储在数据集命名的文件夹内，以分箱数量建立子文件夹，在内部存放不同变量数据的结果，最后将模型的平均结果以`best_param.csv`保存在以数据集命名的文件夹内





## todo

- [ ] 后续进一步完善python代码，将数据预处理过程封装成更加易于使用的包
- [ ] 完善数据填补的代码
- [ ] 完善数据分箱中出现的bug（按照频率分箱出现分位点相同，采用duplicate=False之后出现的bins != labels的问题）
- [ ] 完善shap的包
- [ ] 完善模型的调参优化代码



## 致谢

虽然不是什么惊天动地的大项目，只是一次小练习，但仍收收获满满。熟练了如何应用机器学习模型进行数据处理和建模，同时也总结了一套自己的代码流程。在此期间，离不开yfc，gyw两位师兄的帮助和支持，以及感谢李老师等多位老师在模型效果不佳时给予的帮助和支持。最重要的是还要感谢一同搭档的lys同学，愿一起进步，互相共勉。
