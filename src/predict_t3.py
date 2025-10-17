import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, pairwise_logrank_test, multivariate_logrank_test
import matplotlib.pyplot as plt
from lifelines.plotting import add_at_risk_counts
import sys
from sklearn.preprocessing import KBinsDiscretizer
import argparse
import pickle

parser = argparse.ArgumentParser(prog='predict_t3.py', description='')
#
parser.add_argument('--dataset', dest='inputfile', metavar='INFILE', help='path to csv with entire dataset', default='../data/dataset.csv')
parser.add_argument('--pcr-results', dest='pcrResults', metavar='PCR', help='file with pcr positive-negative patients', default='../results/pcr_ctdnapositive_t3.csv')
parser.add_argument('--t3-cutoff', dest='cutoffPercentile', metavar='CUTOFF', help='percentile of T3 TFE for cut-off', default=70., type=float)
parser.add_argument('--plot', dest='plot', metavar='PLOT', help='whether to make plots for tfe vs covariates', default=False, type=bool)
parser.add_argument('--save-plots', dest='saveplots', metavar='PLOT', help='whether to make plots for tfe vs covariates', default=False, type=bool)
args = parser.parse_args()



data = pd.read_csv(args.inputfile, index_col=0)
data0 = data[data['T0_medseq_success'] == 1]
medianTFE = data0['T0_medseq_TFE'].median()

data = data[data['T3_medseq_success'] == 1]

# 5 T3 deconvolutions were not run originally
# because we had restricted to patients with available BL sample
# these can be run using deconvolve_standard.py

missing = np.where(data['T3_medseq_TFE'].isna())[0]
col = np.where(data.columns == 'T3_medseq_TFE')[0][0]

with open('../results/lcode2tfe_5extraT3s.pkl', 'rb') as f:
    lcode2tfe = pickle.load(f)

for ind in missing:
    lcode = data.iloc[ind]['T3_lcode']
    data.iloc[ind,col] = lcode2tfe[lcode]


data['RFS_days2'] = np.minimum(data['RFS_days'], 365.)

data['T0_medseq_TFE_10'] = 10*data['T0_medseq_TFE']
data['T3_medseq_TFE_100'] = 100*data['T3_medseq_TFE']

m = data['age'].mean()
s = np.std(data['age'], ddof=1)
data['age_std'] = (data['age'] - m ) / s

pcrData = pd.read_csv(args.pcrResults, index_col=0)

pcrPositive = np.zeros(data.shape[0], int)
positives = set(pcrData['MIRACLE_ID'])
for i,p in enumerate(data.index):
    if p in positives:
        pcrPositive[i] = 1

data['pcr_positive'] = pcrPositive



#################################################
cutoff = np.percentile(data['T3_medseq_TFE'], args.cutoffPercentile)
data['T3_MEhigh'] = (data['T3_medseq_TFE'] > cutoff).astype(int)
data['T0_MEhigh'] = (data['T0_medseq_TFE'] > medianTFE).astype(int)

dataP = data[data['T0_VAF_oncomine'] > 0]
dataN = data[data['T0_VAF_oncomine'] == 0]


labs = ['ctDNA low', 'ctDNA high']

fig, ax = plt.subplots(1,1)

cox = CoxPHFitter()
res = cox.fit(data, event_col='Oneyear_RFS_event',duration_col='RFS_days2', formula='~T3_MEhigh')
res.print_summary()
cox.check_assumptions(data)

km = []
cc = ['C8', 'C7']
# for median survival, do not censor at 12 months
for name, grouped_df in data.groupby('T3_MEhigh'):
    kmf = KaplanMeierFitter()
    # kmf.fit(grouped_df["RFS_days2"], grouped_df["Oneyear_RFS_event"], label=name)
    kmf.fit(grouped_df["RFS_days"], grouped_df["RFS_event"], label=labs[name])
    kmf.plot_survival_function(ax=ax, color=cc[name], show_censors=True)
    print('%s: %.1f\n' % (name, kmf.median_survival_time_))
    km.append(kmf)

ax.set_title('Detection of post-operative ctDNA (T3)')
ax.set_xlim(0,366)

mm = 365.25 / 12
ax.set_xticks(np.arange(0,13,2)*mm)
ax.set_xticklabels(np.arange(0,13,2))


add_at_risk_counts(km[0], km[1], ax=ax)
plt.tight_layout()

ax.set_ylabel('Recurrence-Free Survival', fontsize=12)
ax.set_xlabel('time (months)', fontsize=12)



if args.saveplots:
    fig.savefig('../figures/final_KM_ME_T3_%d.pdf' % args.cutoffPercentile, dpi=1200)



fig, ax = plt.subplots(1,2)
for i, (dd,n) in enumerate(zip([dataP, dataN],['oncomine+', 'oncomine-'])):
    print(n)
    cox = CoxPHFitter()
    res = cox.fit(dd, event_col='Oneyear_RFS_event',duration_col='RFS_days2', formula='~T3_MEhigh')
    res.print_summary()
    cox.check_assumptions(dd)

    for name, grouped_df in dd.groupby('T3_MEhigh'):
        kmf = KaplanMeierFitter()
        # kmf.fit(grouped_df["RFS_days2"], grouped_df["Oneyear_RFS_event"], label=name)
        kmf.fit(grouped_df["RFS_days"], grouped_df["RFS_event"], label=labs[name])
        kmf.plot_survival_function(ax=ax[i])
        print('%s: %.1f\n' % (name, kmf.median_survival_time_))
    ax[i].set_title(n)

    print('\n\n\n')



from scipy.stats import spearmanr, fisher_exact
#
print('overlap ctDNA+ mut vs me')
arr = np.array(dataP.groupby(['pcr_positive', 'T3_MEhigh']).count()['hospital']).reshape(2,2)
print(fisher_exact(arr))

plt.show()
#######################################################3
# multi-variable
data = data[~data['fongHigh'].isna()]
data = data[data['T0_medseq_success'] == 1]

dataP = dataP[~dataP['fongHigh'].isna()]
dataP = dataP[dataP['T0_medseq_success'] == 1]


print('\nmv with cutoff')
cox = CoxPHFitter()
res = cox.fit(data,event_col='Oneyear_RFS_event', duration_col='RFS_days2', formula='~age_std + isMale + T3_MEhigh + fongHigh + rightsided_primary + rectal_primary + metachronous + T0_medseq_TFE_10')

print(res.summary.iloc[:,[1,5,6,-2]])
cox.check_assumptions(data)

print('\nmv continuous')
cox = CoxPHFitter()
res = cox.fit(data,event_col='Oneyear_RFS_event', duration_col='RFS_days2', formula='~age_std + isMale + log(T3_medseq_TFE_100) + fongHigh + rightsided_primary + rectal_primary + metachronous + T0_medseq_TFE_10')

print(res.summary.iloc[:,[1,5,6,-2]])
cox.check_assumptions(data)

print('\nmv with cutoff OS')
cox = CoxPHFitter()
res = cox.fit(data,event_col='OS_event', duration_col='OS_days', formula='~age_std + isMale + T3_MEhigh + fongHigh + rightsided_primary + rectal_primary + metachronous + T0_medseq_TFE_10')

print(res.summary.iloc[:,[1,5,6,-2]])
cox.check_assumptions(data)

print('\nmv with cutoff, onco+')
cox = CoxPHFitter()
res = cox.fit(dataP,event_col='Oneyear_RFS_event', duration_col='RFS_days2', formula='~age_std + isMale + T3_MEhigh + fongHigh + rightsided_primary + rectal_primary + metachronous + T0_medseq_TFE_10 + pcr_positive')

print(res.summary.iloc[:,[1,5,6,-2]])
cox.check_assumptions(data)



cc = ['C7', 'C8']
styles = ['-', '--']
name = {(0,0):'BL-/FU-', (0,1):'BL-/FU+', (1,0):'BL+/FU-', (1,1):'BL+/FU+'}
fig,ax = plt.subplots(1,1)
for n, grouped_df in data.groupby(['T0_MEhigh','T3_MEhigh']):
    kmf = KaplanMeierFitter()
    kmf.fit(grouped_df["RFS_days2"], grouped_df["Oneyear_RFS_event"], label=name[n])
    kmf.plot_survival_function(ax=ax, ci_show=False, show_censors=True, color=cc[n[0]], linestyle=styles[n[1]])

    print(kmf.median_survival_time_)

ax.set_title('post-operative ctDNA levels')
if args.saveplots:
    fig.savefig('../figures/KM_T3_cutoff_27percent_RFS.png',dpi=600)


cc = ['C5', 'C4']
styles = ['-', '--']
name = {(0,0):'MUT-/ME-', (0,1):'MUT-/ME+', (1,0):'MUT+/ME-', (1,1):'MUT+/ME+'}
fig,ax = plt.subplots(1,1)
for n, grouped_df in dataP.groupby(['pcr_positive','T3_MEhigh']):
    kmf = KaplanMeierFitter()
    kmf.fit(grouped_df["RFS_days2"], grouped_df["Oneyear_RFS_event"], label=name[n])
    kmf.plot_survival_function(ax=ax, ci_show=False, show_censors=True, color=cc[n[0]], linestyle=styles[n[1]])

    print(kmf.median_survival_time_)

ax.set_title('post-operative ctDNA detection, n=75')
if args.saveplots:
    fig.savefig('../figures/final_zsup_meplusmut_at_t3.eps',dpi=1200)



plt.show()
