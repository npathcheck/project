import pandas as pd

def read(file_path):
    df = pd.read_excel(file_path+"/train.xlsx")
    train_datas = df.values[:,:6]
    train_tags = df.values[:,6]
    train_labels = df.values[:,-1]
    

if __name__ == "__main__":
    read("data/回归")