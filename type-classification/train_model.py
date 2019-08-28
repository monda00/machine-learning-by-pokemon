import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from pandas.io.json import json_normalize
import json

def type_to_zero_one(types, type_to_one):
    if type_to_one in types:
        return 1
    else:
        return 0

def main():
    f = open('../data/pokemon_data.json')
    pokemon_json = json.load(f)
    pokemon_df = json_normalize(data=pokemon_json)

    pokemon_df['type_zero_one'] = pokemon_df['types'].apply(type_to_zero_one, type_to_one='でんき')

    X = pokemon_df.iloc[:, 8:14]
    y = pokemon_df['type_zero_one']

    # データを分ける
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # ロジスティック回帰
    print("Logistic Regression ...")
    lr = LogisticRegression(C=1000)

    # 学習
    print("training ... ")
    lr.fit(X_train, y_train)

    # スコアを表示
    score = lr.score(X_test, y_test)
    print("score is ", score)

if __name__ == '__main__':
    main()

