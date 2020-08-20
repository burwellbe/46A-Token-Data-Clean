import pandas as pd
import json

data = pd.read_csv (r'C:\Users\burwe\Downloads\46A_tokens_labels_2 - main_1.csv')
A = pd.DataFrame(data)
cols = [0,1]
A = A[A.columns[cols]]
df= A.rename(columns={"label_1": "originalLabel"})
df['group_no']= df.T.isnull().all().cumsum()
df['label_source']= "BERT"
df

max_group = max(list(df["group_no"])) + 1

out_json = []

for i in range(max_group):

    df0 = df[df["group_no"] == i]

    list_of_tokens = list(df0["token"])
    list_of_labels = list(df0["originalLabel"])
    source = list(df0["label_source"])[0]

    if i > 0:
        list_of_tokens = list_of_tokens[1:]
        list_of_labels = list_of_labels[1:]

    out_json.append(
        {
            "token": list_of_tokens,
            "originalLabel": list_of_labels,
            "labelSource": source
        }
    )

with open("BERT_data.json", "w") as jfile:
    json.dump(out_json, jfile)

