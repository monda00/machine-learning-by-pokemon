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
    pokemon_df = pd.DataFrame(data=pokemon_json)

    pokemon_df['type_zero_one'] = pokemon_df['types'].apply(type_to_zero_one, type_to_one='でんき')

    print(pokemon_df.head())
    X = json_normalize(data=pokemon_json, record_path='stats')
    y = pokemon_df['type_zero_one']
    print(X.head())

    # データを分ける
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # ロジスティック回帰
    lr = LogisticRegression(C=1000)

    # 学習
    lr.fit(X_train, y_train)

    # スコアを表示
    lr.score(X_test, y_test)

if __name__ == '__main__':
    main()

