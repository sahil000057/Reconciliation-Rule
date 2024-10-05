import pandas as pd
df=pd.read_csv(r"""D:\code\bmc hackathon\Use_case_1_Dataset_sn_hn_try.csv""")
print(df['sn_variant'].nunique())
print(df['hn_variant'].nunique())
print(df['sn_version'].nunique())
print(df['hn_version'].nunique())