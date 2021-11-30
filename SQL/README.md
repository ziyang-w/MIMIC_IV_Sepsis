win+R 打开cmd



```shell
cd /dData\MIMIC_IV

postgres -U postgres -f create.sql

postgres -U postgres -v mimic_data_dir= file_path -f load.sql
```

