import pandas as pd

# 读取文件
df = pd.read_excel('./GFP_data.xlsx')

# print(df)

print(df.head())
print(df.info())

GFP_list = {}

GFP_WT = {'sequence':df.loc[0].aaMutations,
          'GFPtype':df.loc[0].GFPtype,
          'Brightness':df.loc[0].Brightness
         }

print(GFP_WT)

for i in range(10):
    print(df.loc[i].aaMutations)
    print(df.loc[i].GFPtype)
    print(df.loc[i].Brightness)

class load_GFP_list:

