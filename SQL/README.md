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

