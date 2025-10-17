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

parser = argparse.ArgumentParser(prog='predict_t0.py', description='')
#
parser.add_argument('--dataset', dest='inputfile', metavar='INFILE', help='path to csv with entire dataset', default='../data/dataset.csv')
parser.add_argument('--pcr-results', dest='pcrResults', metavar='PCR', help='file with pcr positive-negative patients', default='../results/pcr_ctdnapositive_t3.csv')
parser.add_argument('--t3-cutoff', dest='cutoffPercentile', metavar='CUTOFF', help='percentile of T3 TFE for cut-off', default=70., type=float)
parser.add_argument('--pcr-mutations', dest='mutation_per_patient', metavar='MUTS', help='path to csv file containing which patient has which mutation', default='../data/mutation_per_patient.csv')
parser.add_argument('--plot', dest='plot', metavar='PLOT', help='whether to make plots for tfe vs covariates', default=False, type=bool)
parser.add_argument('--save-plots', dest='saveplots', metavar='PLOT', help='whether to make plots for tfe vs covariates', default=False, type=bool)
args = parser.parse_args()



assaySpecificCutoffs = {
    "TP53p.R273C": 0.205,
    "TP53p.R282W": 0.195,
    "AKT1p.E17K": 0.177,
    "SF3B1p.K700E": 0.318,
    "PIK3CAp.Q546K": 0.312,
    "ERBB3p.E928G": 0.598,
    "ERBB2p.L755S": 0.429,
    "PIK3CAp.N345K": 0.174,
    "PIK3CAp.E726K": 0.04,
    "ESR1p.E380Q": 0.016,
    "ESR1p.D538G": 0.192,
    "KRASp.G12C": 0.018,
    "KRASp.G12V": 0.0161,
    "KRASp.G12D": 0.149,
    "PIK3CAp.E453K": 0.094,
    "PIK3CAp.H1047R": 0.456591,
    "PIK3CAp.H1047L": 0.167728,
    "PIK3CAp.E542K": 0.151616,
    "PIK3CAp.E545K": 0.086578,
    "KRASp.A146T": 0.062055,
    "KRASp.G12S": 0.130409,
    "KRASp.G13D": 0.224034,
    "TP53p.G245S": 0.110684,
    "TP53p.R175H": 0.202985,
    "TP53p.R248Q": 0.181907,
    "TP53p.R248W": 0.081386,
    "TP53p.R273H": 0.150038,
    "BRAFp.V600E": 0.127007,
    "NRASp.Q61R": 0.128255,
    "TP53p.C275Y": 0.373753,
    "KRASmultiplexassay": 0.271134
}



# data = pd.read_csv('../results/t3bigtest.csv', index_col=0)
data = pd.read_csv(args.inputfile, index_col=0)
data0 = data[data['T0_medseq_success'] == 1]
medianTFE = data0['T0_medseq_TFE'].median()

data = data[data['T3_medseq_success'] == 1]

# 5 T3 deconvolutions were not run originally
# because we had restricted to patients with available BL sample
missing = np.where(data['T3_medseq_TFE'].isna())[0]
col = np.where(data.columns == 'T3_medseq_TFE')[0][0]

with open('../results/lcode2tfe_5extraT3s.pkl', 'rb') as f:
    lcode2tfe = pickle.load(f)

for ind in missing:
    lcode = data.iloc[ind]['T3_lcode']
    data.iloc[ind,col] = lcode2tfe[lcode]
#


data['RFS_days2'] = np.minimum(data['RFS_days'], 365.)
print('setting AMP08 OS_event missing variable')
i = np.where(data.index == 'AMP08')[0][0]
j = np.where(data.columns == 'OS_event')[0][0]
data.iloc[i,j] = 0


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

sys.exit(0)


# which mutations
mut = pd.read_csv(args.mutation_per_patient, index_col=0)
mut = mut.iloc[:-1]

mut['patient'] = np.array(pd.Series(mut.index).apply(lambda x: x[:-2]))
mut.set_index('patient', drop=True, inplace=True)


# data['T3_quartile'] = np.digitize(data['T3_medseq_TFE'], np.percentile(data['T3_medseq_TFE'], [0,25,50,75,100]))
if args.plot:
    for n_bins in [2,3,4]:
        print(n_bins)
        kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal')
        resBin = kbd.fit_transform(np.array(data['T3_medseq_TFE']).reshape(-1,1)).reshape(-1,)
        data['T3_quartile'] = resBin

        print(pairwise_logrank_test(data['RFS_days'], data['T3_quartile'], event_observed=data['RFS_event'],t_0=365))

        if n_bins == 2:
            labs = {0: 'bottom 50%', 1: 'top 50%'}
        elif n_bins == 3:
            labs = {0: '0-33%', 1: '33-66%', 2: '66-100%'}
        elif n_bins == 4:
            labs = {0:'0-25%', 1:'25-50%', 2:'50-75%', 3:'75-100%'}
        else:
            raise NotImplementedError

        fig,ax = plt.subplots(1,1)
        for name, grouped_df in data.groupby('T3_quartile'):
            kmf = KaplanMeierFitter()
            # kmf.fit(grouped_df["RFS_days2"], grouped_df["Oneyear_RFS_event"], label=name)
            kmf.fit(grouped_df["RFS_days"], grouped_df["RFS_event"], label=labs[name])
            kmf.plot_survival_function(ax=ax, show_censors=True, ci_show=False)
            print('%s: %.1f\n' % (name, kmf.median_survival_time_))

        ax.set_xlim(0,1100)
        res = multivariate_logrank_test(data['RFS_days'], data['T3_quartile'], event_observed=data['RFS_event'], t_0=365.25)
        print(res)
        print('\n\n')


    # staircase = np.zeros((data.shape[0], int(data['T3_quartile'].max())), int)
    # for i,d in enumerate(data['T3_quartile']):
    #     staircase[i,:int(d)] = 1
    #
    # staircase = pd.DataFrame(data=staircase, index=data.index, columns=['T3_1vs0', 'T3_2vs1', 'T3_3vs2'])
    #
    # data = pd.concat((data, staircase), axis=1)
    kbd = KBinsDiscretizer(n_bins=n_bins, encode='onehot')
    resBin = kbd.fit_transform(np.array(data['T3_medseq_TFE']).reshape(-1,1)).todense()

    onehot = pd.DataFrame(data=resBin[:,1:], index=data.index, columns=['T3_1vs0', 'T3_2vs0', 'T3_3vs0'])

    data = pd.concat((data, onehot), axis=1)

    # plt.show()
    # sys.exit(0)


#################################################
cutoff = np.percentile(data['T3_medseq_TFE'], args.cutoffPercentile)
data['T3_MEhigh'] = (data['T3_medseq_TFE'] > cutoff).astype(int)
data['T0_MEhigh'] = (data['T0_medseq_TFE'] > medianTFE).astype(int)

dataP = data[data['T0_VAF_oncomine'] > 0]
dataN = data[data['T0_VAF_oncomine'] == 0]

# common = np.intersect1d(dataP.index, mut.index)
#
# mut = mut.loc[common]
#
# allassays = set()
# for m in mut['analysed_mutation']:
#     if ';' not in m:
#         allassays.add(m)
#     else:
#         for mm in m.split(';'):
#             allassays.add(mm)
#
# allassays = sorted(list(allassays))
#
# assay2pat = dict()
# for p in mut.index:
#     c = mut.loc[p]['analysed_mutation']
#     if ';' not in c:
#         if c not in assay2pat:
#             assay2pat[c] = [p]
#         else:
#             assay2pat[c].append(p)

pat2assay = dict()
assay2pat = dict()

with open(args.mutation_per_patient, 'r') as f:
    for line in f:
        fields = line.split(',')
        pat = fields[0][:-2]
        if pat not in dataP.index:
            continue

        ass = fields[-1]
        if ';' not in ass:
            ass = ass.rstrip('\n')
            if ass not in assay2pat:
                assay2pat[ass] = [pat]
            else:
                assay2pat[ass].append(pat)
        else:
            for k in ass.split(';'):
                k = k.rstrip('\n')
                if len(k) > 1:
                    if k not in assay2pat:
                        assay2pat[k] = [pat]
                    else:
                        assay2pat[k].append(pat)


for ass, patList in assay2pat.items():
    for pat in patList:
        if pat not in pat2assay:
            pat2assay[pat] = [ass]
        else:
            pat2assay[pat].append(ass)

ntested = []
nvafpos = []
nmepos = []
assayNames = []
intersection = []
jaccard = []
for assay in assay2pat:
    print('Assay: %s, %d patients tested' % (assay, len(assay2pat[assay])))


    assayNames.append(assay)
    ntested.append(len(assay2pat[assay]))

    tmp = dataP.loc[assay2pat[assay]]

    mepos = tmp[tmp['T3_MEhigh']==1].index
    mutpos = tmp[tmp['pcr_positive']==1].index

    nvafpos.append(mutpos.shape[0])
    nmepos.append(mepos.shape[0])

    intersection.append(np.intersect1d(mepos, mutpos).shape[0])



comparison = pd.DataFrame({'nr_patients': ntested, 'vaf_pos': nvafpos, 'me_pos': nmepos, 'intersection': intersection}, index=assayNames)


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

res = multivariate_logrank_test(data['RFS_days'], data['T3_MEhigh'], event_observed=data['RFS_event'], t_0=365.25)
print(res.p_value)



# plt.show()
# sys.exit(0)


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


# Nb = 100
# pp = np.zeros(Nb)
# for i in range(Nb):
#     ind = np.random.choice(np.arange(dataP.shape[0]), size=dataN.shape[0], replace=True)
#     dd = dataP.iloc[ind]
#     cox = CoxPHFitter()
#     res = cox.fit(dd, event_col='Oneyear_RFS_event',duration_col='RFS_days2', formula='~T3_MEhigh')
#     pp[i] = res.summary['p'][0]

from lifelines.statistics import power_under_cph
# of oncomine- at T3
# 17 MEhigh, 25 low
# high group has 75% chance of 1-year recurrence
# low 46%, based on Lissa's paper
# for a HR of 2 and an alpha of 5%
print(power_under_cph(17,25, 0.75, 0.46, 2.0, 0.05))

from scipy.stats import spearmanr, fisher_exact
#
print('overlap ctDNA+ mut vs me')
arr = np.array(dataP.groupby(['pcr_positive', 'T3_MEhigh']).count()['hospital']).reshape(2,2)
print(fisher_exact(arr))

plt.show()
sys.exit(0)
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

ind = np.where(np.isfinite(data['HGP_fully_desmoplastic']))[0]
print('\nmv with cutoff + HGP')
cox = CoxPHFitter()
res = cox.fit(data.iloc[ind],event_col='Oneyear_RFS_event', duration_col='RFS_days2', formula='~age_std + isMale + T3_MEhigh + fongHigh + rightsided_primary + rectal_primary + metachronous + T0_medseq_TFE_10 + HGP_fully_desmoplastic')

print(res.summary.iloc[:,[1,5,6,-2]])
cox.check_assumptions(data)

ind = np.where(np.isfinite(data['HGP_fully_desmoplastic']))[0]
print('\nmv with cutoff + HGP - OS')
cox = CoxPHFitter()
res = cox.fit(data.iloc[ind],event_col='OS_event', duration_col='OS_days', formula='~age_std + isMale + T3_MEhigh + fongHigh + rightsided_primary + rectal_primary + metachronous + T0_medseq_TFE_10 + HGP_fully_desmoplastic')

print(res.summary.iloc[:,[1,5,6,-2]])
cox.check_assumptions(data)

# name = {0:'ctDNA-', 1:'ctDNA+'}
# fig,ax = plt.subplots(1,1)
# for n, grouped_df in data.groupby('T3_MEhigh'):
#     kmf = KaplanMeierFitter()
#     kmf.fit(grouped_df["RFS_days2"], grouped_df["Oneyear_RFS_event"], label=name[n])
#     kmf.plot_survival_function(ax=ax, show_censors=True)
#
# ax.set_title('post-operative ctDNA levels')
# if args.saveplots:
#     fig.savefig('../figures/KM_T3_cutoff_27percent_RFS.png',dpi=600)

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
