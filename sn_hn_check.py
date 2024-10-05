import pandas as pd
import uuid

def get_uuid_version(uuid_str):
    try:
        if isinstance(uuid_str, str):  
            u = uuid.UUID(uuid_str)  
            return u.version
        else:
            return None
    except ValueError:
        return None 

def get_uuid_variant(uuid_str):
    try:
        if isinstance(uuid_str, str): 
            u = uuid.UUID(uuid_str)
            return u.variant
        else:
            return None
    except ValueError:
        return None 

df=pd.read_csv(r"""D:\code\bmc hackathon\Use_case_1_Dataset.csv""") 
df=df.filter(['serialnumber','hostname','ReconciliationRuleId'])
df['sn_version'] = df['serialnumber'].apply(get_uuid_version)
df['sn_variant'] = df['serialnumber'].apply(get_uuid_variant)

df['hn_version'] = df['hostname'].apply(get_uuid_version)
df['hn_variant'] = df['hostname'].apply(get_uuid_variant)

df.to_csv(r"""D:\code\bmc hackathon\Use_case_1_Dataset_sn_hn_try.csv""", index=False) 

print(df[['serialnumber', 'sn_version', 'sn_variant', 'hostname', 'hn_version', 'hn_variant']].head())
