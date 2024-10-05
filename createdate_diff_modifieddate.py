import pandas as pd
import numpy as np

def diff(a, b):
    return a - b

df = pd.read_csv(r"""D:\code\bmc hackathon\Use_case_1_Dataset.csv""")
df = df.filter(['createdate', 'modifieddate', 'ReconciliationRuleId'])
df['Diff'] = df.apply(lambda x: diff(x['modifieddate'], x['createdate']), axis=1)

df.to_csv(r"""D:\code\bmc hackathon\Use_case_1_Dataset_diff.csv""", index=False) 
df2 = pd.read_csv(r"""D:\code\bmc hackathon\Use_case_1_Dataset_diff.csv""")

# print("------------cov-----------")
# covariance_matrix = df2.cov()

# if 'ReconciliationRuleId' in covariance_matrix.columns:
#     covariance_with_target = covariance_matrix['ReconciliationRuleId']
#     sorted_covariances = covariance_with_target.abs().sort_values(ascending=False)
#     print(sorted_covariances.head(10))
# else:
#     print("'ReconciliationRuleId' not found in covariance matrix")

print("------------corr---------")
correlation_matrix = df2.corr()

if 'ReconciliationRuleId' in correlation_matrix.columns:
    correlation_with_target = correlation_matrix['ReconciliationRuleId']
    sorted_correlations = correlation_with_target.abs().sort_values(ascending=False)
    print(sorted_correlations.head(10))
else:
    print("'ReconciliationRuleId' not found in correlation matrix")
