import lifelines
from dcurves import *
from sklearn.model_selection import StratifiedKFold
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(prog='doDca.py', description='')

parser.add_argument('--dataset', dest='inputfile', metavar='INFILE', help='path to csv with entire dataset', default='../data/dataset.csv')
parser.add_argument('--outcome', dest='surv', metavar='outcome', help='OS or RFS', default='OS', type=str)
parser.add_argument('--years', dest='years', metavar='YEARS', help='timepoint to evaluate', default=3.0, type=float)
parser.add_argument('--postop', dest='postop', metavar='POSTOP', help='whether to include postoperative', default=0, type=int)
parser.add_argument('--nfolds', dest='folds', metavar='NFOLDS', help='# cv folds', default=10, type=int)
args = parser.parse_args()


assert args.surv == 'OS' or args.surv == 'RFS'

data = pd.read_csv(args.inputfile, index_col=0)

# for 1-year specifically
data['RFS_days2'] = np.minimum(data['RFS_days'], 365.0)

dataME = data[data['T0_medseq_success'] == 1.0]

# AMP08 had recurrence and then no further follow-up
# for now censor at day of recurrence
print('!!!!!!!!!!!!!!!WARNING!!!!!!!!!!!')
print('setting AMP08 OS_event missing variable')
i = np.where(dataME.index == 'AMP08')[0][0]
j = np.where(dataME.columns == 'OS_event')[0][0]
dataME.iloc[i,j] = 0

medianTFE = dataME['T0_medseq_TFE'].median()

print(dataME.shape)


if args.postop:
    dataME = dataME[dataME['T3_medseq_success'] == 1]
    cutoffT3 = np.percentile(dataME['T3_medseq_TFE'], 70)
    dataME['T3_ctdna_hi'] = (dataME['T3_medseq_TFE'] > cutoffT3).astype(int)


print(dataME.shape)


modality = 'medseq'
dataME['ctDNAhigh_%s' % modality] = (dataME['T0_%s_TFE' % modality] > medianTFE).astype(int)

# multiply by 100 and divdie by 10
dataME['T0_%s_TFE_10' % modality] = 10. * dataME['T0_%s_TFE' % modality]
# this way, increase of 1 in the variable --> increase of 10% in tf
# this makes interpretation of HRs a bit easier



# standardize age to 0 mean and unit std
m = dataME['age'].mean()
s = np.std(dataME['age'], ddof=1)
dataME['age_std'] = (dataME['age'] - m ) / s



dataME_mv = dataME[~dataME['fongHigh'].isna()]

targetDays = 365.25 * args.years

osEventAtTarget = np.zeros(dataME_mv.shape[0], int)
for i in range(dataME_mv.shape[0]):
    if dataME_mv.iloc[i]['%s_days' % args.surv] > targetDays:
        osEventAtTarget[i] = 0
    else:
        osEventAtTarget[i] = dataME_mv.iloc[i]['%s_event' % args.surv]


cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

risks = np.zeros(dataME_mv.shape[0])

if args.postop:
    myformula = 'T0_medseq_TFE_10 + T3_ctdna_hi + rightsided_primary + rectal_primary + fongHigh + age_std + isMale + metachronous'
else:
    myformula = 'T0_medseq_TFE_10 + rightsided_primary + rectal_primary + fongHigh + age_std + isMale + metachronous'

myformula = 'rightsided_primary + rectal_primary + fongHigh + age_std + isMale + metachronous'

concordanceIndex = np.zeros((2,args.folds))

for i, (trainInd, testInd) in enumerate(cv.split(dataME_mv, osEventAtTarget)):
    Xtrain = dataME_mv.iloc[trainInd]
    Xtest = dataME_mv.iloc[testInd]

    cph = lifelines.CoxPHFitter()
    res = cph.fit(Xtrain, duration_col='%s_days' % args.surv, event_col='%s_event' % args.surv, formula=myformula)
    res.print_summary()

    cph_pred_vals = cph.predict_survival_function(Xtest, times=[targetDays])

    risks[testInd] = [1 - val for val in cph_pred_vals.iloc[0, :]]
    concordanceIndex[0, i] = cph.score(Xtrain, scoring_method='concordance_index')
    concordanceIndex[1, i] = cph.score(Xtest, scoring_method='concordance_index')


dataME_mv['pr_death_3y'] = risks


dcares = dca(data=dataME_mv, outcome='%s_event' % args.surv, modelnames=['pr_death_3y', 'fongHigh'], thresholds=np.arange(0,0.5,0.005), time=targetDays, time_to_outcome_col='%s_days' % args.surv)

plot_graphs(dcares, graph_type='net_benefit', smooth_frac=0.5, file_name='delete.png')

if args.postop:
    prefix = 'postop'
else:
    prefix = 'preop'
filename = '../results/%s_dca_%s_at_%.1f_years.csv' % (prefix, args.surv, args.years)

# dcares.to_csv(filename)

plt.close()
