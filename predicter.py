import numpy as np
import pandas as pd
import pickle

pokemons = pd.read_csv("pokedex.csv")

pokemons['Type 2'].replace((np.nan), ('NoType'), inplace=True)
type_1 = pokemons['Type 1'].unique()
type_2 = pokemons['Type 2'].unique()

type_2 = np.delete(type_2, np.where(type_2 == 'NoType'))

pokemon_types = pd.Series(type_1).append(pd.Series(type_2)).unique()

mlps = []
for i in pokemon_types:
    mlps.append(pickle.load(open(f"ultra_good_models/{i}.sav", 'rb')))

parms = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']

get_parms = []

print('Insert the features:')

for i in parms:
    parm = ""
    while not parm.isnumeric():
        parm = input(f'      {i}: ')
        if parm.isnumeric():
            get_parms.append(int(parm))
        else:
            print('The features must be integer values, retry')

is_legendary = ""
while is_legendary != 'y' and is_legendary != 'n':
    is_legendary = input('Is this pokemon legendary? [Y/N]\n').lower()
    if is_legendary == 'y':
        get_parms.append(1)
    elif is_legendary == 'n':
        get_parms.append(0)
    else:
        print('Choice not allowed!')

to_predict = [get_parms]

predictions = []
for index, mlp in enumerate(mlps):
    pred = mlp.predict(to_predict)
    predictions.append([pred, pokemon_types[index]])

print('Results:')
found = False
for index, pred in enumerate(predictions):
    if pred[0] == 1:
        found = True
        print(pred[1])
    elif index == len(predictions) - 1 and not found:
        print('No pokemon types found')
