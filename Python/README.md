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