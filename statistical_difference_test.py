import pandas as pd
import numpy as np
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
import random

df = pd.read_csv('./data/features/features.csv')
df.drop(columns=['Unnamed: 0'], axis = 1, inplace=True)
labels = np.load('./data/features/labels.npy')
df['label'] = pd.Series(labels)

df['label'].replace({0:'c', 1:'a', 2:'f'}, inplace=True)
cols = df.columns

features = [c for c in cols if c != 'label']
formula = ' + '.join(features)

formula += ' ~ label'

manova_model = MANOVA.from_formula(formula=formula, data=df)

manova_results = manova_model.mv_test()
print(manova_results)

good_feature = 'hjorth_activity_alpha_9'
mid_feature = 'hjorth_activity_beta_18'

'''posthocs = []


for f in good_feature:
    posthoc = sp.posthoc_ttest(df, val_col=f, group_col='label', p_adjust = 'holm')
    posthocs.append(np.array(posthoc))

num_significant_ac = 0
num_significant_af = 0
num_significant_fc = 0

for p in posthocs:
    if p[0][1] < 0.05: # A C
        num_significant_ac += 1
        #print(f"A i C: {p[0][1]}")

    if p[0][2] < 0.05: # A F
        num_significant_af += 1
        print(f"A i F: {p[0][2]}")
    
    if p[1][2] < 0.05: # C F
        num_significant_fc += 1
        #print(f"C i F: {p[1][2]}")

print(f'Alzheimer & Controlled: {num_significant_ac}/{len(features)}')
print(f'Alzheimer & FTD: {num_significant_af}/{len(features)}')
print(f'FTD & Controlled: {num_significant_fc}/{len(features)}')

#'hjorth_activity_alpha_9' '''

random.shuffle(features)

num_significant_ac = 0
num_significant_af = 0
num_significant_cf = 0

for f in features:
    tukey = pairwise_tukeyhsd(endog=df[f], groups=df['label'], alpha = 0.05)
    reject = [int(x) for x in tukey.reject]
    num_significant_ac += reject[0]
    num_significant_af += reject[1]
    num_significant_cf += reject[2]

print(f"AC: {num_significant_ac}")
print(f"AF: {num_significant_af}")
print(f"CF: {num_significant_cf}")

