import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def import_pokemon_data():
    df = pd.read_json('../data/pokemon_data.json')

    return df

def type_to_zero_one(types, type_to_one):
    if type_to_one in types:
        return 1
    else:
        return 0

def main():
    pokemon_df = import_pokemon_data()

    pokemon_df['type_zero_one'] = pokemon_df['types'].apply(type_to_zero_one, type_to_one='でんき')


if __name__ == '__main__':
    main()

