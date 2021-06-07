import pandas as pd
array_dict = []
def only_tokens(row):
    global array_dict

    row = row.values.tolist()
    tokens = row[0].split(" ")
    syllables = row[1].split(" ")
        # for token, syllable in zip(tokens, syllables):
    if len(tokens) == len(syllables):
        for token, syllable in zip(tokens, syllables):
            if (syllable != '0'): array_dict.append((token, syllable))

def r(df, cols):
    for col in cols:
        df[col] = df[col].apply(lambda u: str(u).encode('ascii', 'ignore'))
        df[col] = df[col].str.decode('utf-8').fillna(df[col]) 
    return df

df1 = pd.read_csv('datasets\syllable_word_browns.csv')
df2 = pd.read_csv('datasets\syllable_word_wordnet_1.csv')
df3 = pd.read_csv('datasets\syllable_word_wordnet_2.csv')
df4 = pd.read_csv('datasets\syllable_word_wordnet_3.csv')
df5 = pd.read_csv('datasets\syllable_word_wordnet_4.csv')
df6 = pd.read_csv('datasets\syllable_word.csv')

df = pd.concat([df1, df2, df3, df4, df5, df6])
df = r(df, ['token', 'syllable'])
df.reset_index(inplace=True, drop=True)

df = df[df['syllable'] != '0']
df.reset_index(inplace=True, drop=True)

df.apply(only_tokens, axis=1)

df_ = pd.DataFrame(array_dict, columns=['token', 'syllable'])
df_.to_csv('datasets/syllables.csv')

df_test = df_[:int(df_.shape[0]*0.2)].reset_index(drop=True)
df_train = df_[int(df_.shape[0]*0.2):int(df_.shape[0]*0.9)].reset_index(drop=True)
df_dev = df_[int(df_.shape[0]*0.9):].reset_index(drop=True)

df_test.to_csv('data/custom_syllables/test.csv')
df_train.to_csv('data/custom_syllables/train.csv')
df_dev.to_csv('data/custom_syllables/dev.csv')