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
min_loss_curve = []
for index, mlp in enumerate(mlps):
    pred = mlp.predict(to_predict)
    min_loss_curve.append(min(mlp.loss_curve_))
    predictions.append(pred)

results = []
for index, pred in enumerate(predictions):
    if pred == 1:
        results.append(index)
min_loss_curve_of_results = {}
for index in results:
    min_loss_curve_of_results.update({index: min_loss_curve[index]})

best_results = list(dict(sorted(min_loss_curve_of_results.items(), key=lambda item: item[1])).keys())[:2]
for index in best_results:
    print(pokemon_types[index])

if not np.any(predictions):
    print('No pokemon types found')
