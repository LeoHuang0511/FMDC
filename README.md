1. For colorization pretrian, please modify the config.py __C.task = "LAB", you can see more details in config.py

2. After colorization, change __C.task = "DEN" and run python train.py to adapt the model on inflow\outflow count, __C.DATASET can change the training dataset.

3. Finally, use the test_HT21.py and test_CARLA to test the model.

4. Our CARLA dataset can be downoaded from here https://nycu1-my.sharepoint.com/:u:/g/personal/s311505011_ee11_m365_nycu_edu_tw/ESv4FNy51fJEiCfByfgOea0B5yxE4JZh4JmTTGtEGTdLQw?e=UlorCN.

5. The pretrained weight for HT21 can download from https://nycu1-my.sharepoint.com/personal/s311505011_ee11_m365_nycu_edu_tw/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fs311505011%5Fee11%5Fm365%5Fnycu%5Fedu%5Ftw%2FDocuments%2FWACV24%5FPeople%5FFlow%2Fep%5F13%5Fiter%5F33000%5Fmae%5F13%2E118%5Fmse%5F13%2E494%5Fseq%5FMAE%5F0%2E237%5FWRAE%5F0%2E273%5FMIAE%5F2%2E962%5FMOAE%5F1%2E968%2Epth&parent=%2Fpersonal%2Fs311505011%5Fee11%5Fm365%5Fnycu%5Fedu%5Ftw%2FDocuments%2FWACV24%5FPeople%5FFlow&p=14
