import pandas as pd
from glob import glob


excel_dir = "G://Keyence/ExcelSheets/"
save_file = "G://Keyence/ExcelSheets/bold-master-data.csv"

excel_lst = glob(excel_dir + "ForStefan*")
master_df = pd.read_excel(excel_lst[0])

for excel in excel_lst[1:]:
    print(master_df)
    print(master_df.shape)
    df = pd.read_excel(excel)
    master_df = master_df.append(df, ignore_index=True)

master_df.to_csv(save_file, index=False)