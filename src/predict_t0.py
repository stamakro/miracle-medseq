import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test
from lifelines.calibration import survival_probability_calibration
from statsmodels.formula.api import logit
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, precision_recall_curve, roc_curve
from sklearn.calibration import calibration_curve, CalibrationDisplay
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.multitest import multipletests
from scipy.stats import mannwhitneyu, kruskal, spearmanr
import sys
import sklearn
import warnings
from betakde import *
warnings.filterwarnings('ignore', category=sklearn.exceptions.UndefinedMetricWarning)

parser = argparse.ArgumentParser(prog='predict_t0.py', description='')

parser.add_argument('--dataset', dest='inputfile', metavar='INFILE', help='path to csv with entire dataset', default='../data/dataset.csv')
parser.add_argument('--miraclecfdnainfo', dest='medseq_sample_info', metavar='INFOMEDSEQ', help='path to csv file containing CpG reads, mapping etc for MIRACLE cfdna samples', default='/home/stavros/Desktop/code/nb-deconv/medseq-deconv/data/new_sampleoverzicht_SW_20240219.csv')
parser.add_argument('--oncominefile', dest='oncomine_file', metavar='INFOONCOMINE', help='path to csv file containing mutation data', default='/home/stavros/emc/users/smakrodimitris/miracle/src/miracle-meth/data/mutation_data_pergene_oncomine_medseq_cohort.csv')
parser.add_argument('--cnv-file', dest='cnvfile', metavar='CNVTFE', help='CNV tfe file', default='../data/medseqcnv_tumor_fraction_estimates_n120.csv', type=str)
parser.add_argument('--n-bootstraps', dest='nBootstraps', metavar='BOOTSTRAPS', help='how many bootstraps for feature importance', default=1000, type=int)
parser.add_argument('--n-permutations', dest='nPermutations', metavar='PERMUTATIONS', help='how many permutations to establish significance of performance', default=1000, type=int)
parser.add_argument('--n-folds', dest='nFolds', metavar='CVFOLDS', help='how many folds in stratified CV', default=3, type=int)
parser.add_argument('--plot', dest='plot', metavar='PLOT', help='whether to make plots for tfe vs covariates', default=False, type=bool)
parser.add_argument('--save-plots', dest='saveplots', metavar='PLOT', help='whether to make plots for tfe vs covariates', default=False, type=bool)
args = parser.parse_args()



data = pd.read_csv(args.inputfile, index_col=0)

# for 1-year specifically
data['RFS_days2'] = np.minimum(data['RFS_days'], 365.0)

dataME = data[data['T0_medseq_success'] == 1.0]

info = pd.read_csv(args.medseq_sample_info, index_col=0 )
# by doing this, the order is correct
info = info.loc[dataME['T0_lcode']]

info['Mapping'] = 100. * info['Used reads'] / info['CpG reads']

dataOnco = pd.read_csv(args.oncomine_file, index_col=0)
dataOnco = dataOnco.loc[dataME.index]

dataME = pd.concat((dataME, dataOnco), axis=1)

cnvtfe = pd.read_csv(args.cnvfile, index_col=0)['tumorFractionEstimate'].to_dict()

dataME['T0_cnv_TFE'] = dataME['T0_lcode'].map(cnvtfe)

pat2lcode = dataME['T0_lcode'].to_dict()
lcode2pat = dict()
for k,v in pat2lcode.items():
    lcode2pat[v] = k

dataME = pd.concat((info.rename(index=lcode2pat), dataME), axis=1)

dataME.rename(columns={'T0_VAF_oncomine': 'T0_vaf_TFE'}, inplace=True)


zeroIndCnv = np.where(dataME['T0_cnv_TFE'] == 0)[0]
nonzeroIndCnv = np.where(dataME['T0_cnv_TFE'] != 0)[0]

modalities = ['medseq', 'cnv', 'vaf']

if args.plot:
    for ii, modality in enumerate(modalities):
        feature2pvalue = dict()
        feature2type = dict()

        feature2pvalueZero = dict()

        logg = [True, False, False, False, False, True]
        for i, c in enumerate(['CpG reads', '% filtered reads', 'cfdna_yield', 'Mapping', 'age', 'replacement']):
            # print(c)
            feature2type[c] = 'continuous'
            if c == 'replacement':
                # remove some missing
                iind = np.where(~dataME['replacement'].isna())[0]
                r, p = spearmanr(dataME.iloc[iind]['T0_%s_TFE' % modality], dataME.iloc[iind][c])
            else:
                r, p = spearmanr(dataME['T0_%s_TFE' % modality], dataME[c])

            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.scatter(dataME[c], dataME['T0_%s_TFE' % modality])
            ax.set_xlabel(c)
            ax.set_ylabel('estimated tumor fraction')
            if logg[i]:
                ax.set_xscale('log')

            ax.set_title(r'$\rho = %.2f$' % r)
            feature2pvalue[c] = p

            if modality == 'cnv':
                feature2pvalueZero[c] = mannwhitneyu(dataME.iloc[zeroIndCnv][c], dataME.iloc[nonzeroIndCnv][c])

            if args.saveplots:
                fig.savefig('../figures/final_zsup_correlation_tfe_%s_%s.png' % (modality, c), dpi=1200)
            plt.close()

        # binary
        extraColsToTest = ['max1y_primary_CRLM', 'isN0', 'Two_or_more_CRLMs', 'CEA_veryhigh', 'diameter5plus',
        'isR1', 'fongHigh', 'isMale', 'metachronous', 'APC_mutant', 'TP53_mutant', 'KRAS_mutant', 'PIK3CA_mutant']
        pp = np.zeros(len(extraColsToTest), float)

        for i,c in enumerate(extraColsToTest):
            # print(c)
            feature2type[c] = 'binary'
            batchVAFs = [dataME.iloc[groupind][('T0_%s_TFE' % modality)] for batch, groupind in dataME.groupby(c).indices.items()]
            stat, p = mannwhitneyu(*batchVAFs)
            # print('%.5f' % p)
            pp[i] = p
            fig, ax = plt.subplots(1,1)
            sns.boxplot(dataME, x=c, y=('T0_%s_TFE' % modality), ax=ax)

            #ax.set_title('p-value = %.2f' % p)
            ax.set_ylabel('estimated tumor fraction')

            if modality == 'cnv':
                feature2pvalueZero[c] = mannwhitneyu(dataME.iloc[zeroIndCnv][c], dataME.iloc[nonzeroIndCnv][c])

            if args.saveplots:
                fig.savefig('../figures/final_zsup_correlation_tfe_%s_%s.png' % (modality,c), dpi=1200)
            plt.close()
            feature2pvalue[c] = p


        extraColsToTest = ['MeD-seq batch', 'hospital']
        for c in extraColsToTest:
            batchVAFs = [dataME.iloc[groupind][('T0_%s_TFE' % modality)] for batch, groupind in dataME.groupby(c).indices.items()]
            fig, ax = plt.subplots(1,1)
            sns.boxplot(dataME, x=c, y=('T0_%s_TFE' % modality))
            ax.set_xlabel(c)
            ax.set_ylabel('tumor fraction estimate')

            p = kruskal(*batchVAFs)[1]
            # ax.set_title('p-value = %.2f' % p)

            feature2pvalue[c] = p
            feature2type[c] = 'categorical'
            if args.saveplots:
                fig.savefig('../figures/final_zsup_correlation_tfe_%s_%s.png' % (modality, c), dpi=1200)
            plt.close()

        plt.close('all')

        if ii == 0:
            pvals = pd.Series(feature2pvalue)
            pvals = pd.DataFrame(pvals).rename(columns={0: 'p-value'})
            pvals['type'] = pvals.index.map(feature2type)

            pcorDict = dict()
            for cc in ['binary', 'continuous', 'categorical']:
                pvalsSubset = pvals[pvals['type'] == cc]
                ppcor = multipletests(pvalsSubset['p-value'], method='fdr_bh')[1]
                for feat, p in zip(pvalsSubset.index, ppcor):
                    pcorDict[feat] = p

            pvals['FDR_%s' % modality] = pvals.index.map(pcorDict)
        else:
            pvals2 = pd.Series(feature2pvalue)
            pvals2 = pd.DataFrame(pvals2).rename(columns={0: 'p-value'})
            pvals2['type'] = pvals2.index.map(feature2type)

            pcorDict = dict()
            for cc in ['binary', 'continuous', 'categorical']:
                pvalsSubset = pvals2[pvals2['type'] == cc]
                ppcor = multipletests(pvalsSubset['p-value'], method='fdr_bh')[1]
                for feat, p in zip(pvalsSubset.index, ppcor):
                    pcorDict[feat] = p

            pvals['FDR_%s' % modality] = pvals.index.map(pcorDict)

    print(pvals)

# AMP08 had recurrence and then no further follow-up
# for now censor at day of recurrence
print('!!!!!!!!!!!!!!!WARNING!!!!!!!!!!!')
print('setting AMP08 OS_event missing variable')
i = np.where(dataME.index == 'AMP08')[0][0]
j = np.where(dataME.columns == 'OS_event')[0][0]
dataME.iloc[i,j] = 0



# median follow-up
print('Median follow-up')
kmf = KaplanMeierFitter()
res = kmf.fit(dataME['RFS_days'], event_observed=1-dataME['RFS_event'])
d = kmf.median_survival_time_
print('RFS: %.1f days = %.1f months' % (d, 12*d/365.25))


kmf = KaplanMeierFitter()
res = kmf.fit(dataME['OS_days'], event_observed=1-dataME['OS_event'])
d = kmf.median_survival_time_
print('OS: %.1f days = %.1f months' % (d, 12*d/365.25))

sys.exit(0)

clinicalVAFpositive = dataME[dataME['T0_vaf_TFE'] > 0]
name = {'medseq': 'methylation', 'cnv': 'ichorCNA', 'vaf': 'VAF'}
if args.plot:
    for modality in ['medseq', 'cnv']:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        ax.scatter(clinicalVAFpositive['T0_vaf_TFE'], clinicalVAFpositive['T0_%s_TFE' % modality])

        M = np.maximum(np.max(clinicalVAFpositive['T0_vaf_TFE']), np.max(clinicalVAFpositive['T0_%s_TFE' % modality]))
        xx = np.linspace(0, M, 4)
        ax.plot(xx, xx, color='k', linestyle='--')

        ax.set_xlabel('Variant allele frequency', fontsize=15)
        ax.set_ylabel('%s tumor fraction estimate' % name[modality], fontsize=18)

        ax.set_xlim(-0.01, M+0.01)
        ax.set_ylim(-0.01, M+0.01)

        rhoS, pS = spearmanr(clinicalVAFpositive['T0_vaf_TFE'], clinicalVAFpositive['T0_medseq_TFE'])
        # rhoP, pP = pearsonr(clinicalVAFpositive['T0_vaf_TFE'], clinicalVAFpositive['T0_medseq_TFE'])

        print('T0, oncomine (N=%d):' % len(clinicalVAFpositive))
        # print('Pearson: %.2f\t%.5f' % (rhoP, pP))
        print('Spearman: %.2f\t%.5f' % (rhoS, pS))
        print('\n\n')

        ax.set_title('n=%d, rho = %.2f, p<1e-6' % (len(clinicalVAFpositive), rhoS))
        fig.savefig('../figures/final_vaf_tfe_%s_correlation.png' % modality, dpi=1200)
        # fig.savefig('../figures/final_vaf_tfe_correlation.svg', dpi=600)

        ii = np.where(dataME['T0_vaf_TFE'] > 0)[0]
        jj = np.setdiff1d(np.arange(dataME.shape[0]), ii)

        xx = np.linspace(0,0.5, 51)
        densop = beta_kde(dataME.iloc[ii]['T0_%s_TFE' % modality], 0.04)
        denson = beta_kde(dataME.iloc[jj]['T0_%s_TFE' % modality], 0.04)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(xx, densop(xx), color='C4', label='mutation positive')
        ax.plot(xx, denson(xx), color='C5', label='mutation negative')

        ax.set_xlabel('%s TFE' % name[modality], fontsize=14)
        ax.set_ylabel('probability density', fontsize=14)

        ax.legend()

        fig.savefig('../figures/final_tfe_%s_vafpos_vafneg.png' % modality, dpi=1200)
        # fig.savefig('../figures/final_tfe_vafpos_vafneg.svg', dpi=600)

sys.exit(0)
medianTFE = dataME['T0_medseq_TFE'].median()
print('methylation cohort, N=%d' % dataME.shape[0])
print('median TFE: %.3f' % medianTFE)

for modality in modalities:
    dataME['ctDNAhigh_%s' % modality] = (dataME['T0_%s_TFE' % modality] > medianTFE).astype(int)

    # multiply by 100 and divdie by 10
    dataME['T0_%s_TFE_10' % modality] = 10. * dataME['T0_%s_TFE' % modality]
    # this way, increase of 1 in the variable --> increase of 10% in tf
    # this makes interpretation of HRs a bit easier

    # the same number in log scale, with pseudo count to deal with zeros in ichorcna and vaf
    dataME['T0_%s_TFE_log' % modality] = np.log(1e-6 + dataME['T0_%s_TFE_10' % modality])



# standardize age to 0 mean and unit std
m = dataME['age'].mean()
s = np.std(dataME['age'], ddof=1)
dataME['age_std'] = (dataME['age'] - m ) / s


mm = []
surv = []
modeling = []
unimulti = []
metric = []
metricType = []

T = dataME['RFS_days2']
E = dataME['Oneyear_RFS_event']

dataME_mv = dataME[~dataME['fongHigh'].isna()]


for modality in modalities:
    print('\n\n\n')
    print(modality)


    fig = plt.figure()
    ax =fig.add_subplot(111)

    dem = (dataME['ctDNAhigh_%s' % modality] == 1)

    kmfH = KaplanMeierFitter()
    kmfH.fit(T[dem], event_observed=E[dem], label='ctDNAhigh')
    kmfH.plot_survival_function(ax=ax, show_censors=True, color='C7', ci_show=True)

    print('ctDNA high, median survival %f days' % kmfH.median_survival_time_)

    kmfL = KaplanMeierFitter()
    kmfL.fit(T[~dem], event_observed=E[~dem], label='ctDNAlow')
    kmfL.plot_survival_function(ax=ax, show_censors=True, color='C8', ci_show=True, loc=slice(0,365))
    print('ctDNA low, median survival %f days' % kmfL.median_survival_time_)

    ax.set_xticks(np.linspace(0,365,7))
    ax.set_xticklabels(np.arange(0,13,2))
    ax.set_xlabel('time (months)', fontsize=16)

    ax.set_ylabel('Recurrence-free survival')
    ax.axvline(365., color='k', linestyle='--')
    results = logrank_test(T[dem], T[~dem], E[dem], E[~dem])
    results.print_summary()

    ax.set_title(name[modality])
    add_at_risk_counts(kmfH, kmfL, ax=ax)
    plt.tight_layout()


    cox = CoxPHFitter()
    res = cox.fit(dataME, duration_col='RFS_days2', event_col='Oneyear_RFS_event', formula=('~ ctDNAhigh_%s' % modality))

    print('HR = %.2f, 95%% CI = [%.2f, %.2f]' % (res.summary['exp(coef)'], res.summary['exp(coef) lower 95%'], res.summary['exp(coef) upper 95%']))
    mm.append(modality)
    surv.append('RFS-1y')
    modeling.append('cutoff')
    unimulti.append('uni')
    metric.append( res.summary['exp(coef)'].iloc[0])
    metricType.append('HR')

    mm.append(modality)
    surv.append('RFS-1y')
    modeling.append('cutoff')
    unimulti.append('uni')
    metric.append( res.summary['p'].iloc[0])
    metricType.append('p-value')
    if args.saveplots:
        fig.savefig('../figures/KM_cfDNAhighlow_T0_%s_RFS2.png' % modality, dpi=600)


    # univariate continuous
    print('continuous, by 10')
    cph = CoxPHFitter()
    res = cph.fit(dataME, duration_col='RFS_days2', event_col='Oneyear_RFS_event', formula=('T0_%s_TFE_10' % modality))
    res.print_summary()

    mm.append(modality)
    surv.append('RFS-1y')
    modeling.append('continuous')
    unimulti.append('uni')
    metric.append( res.summary['exp(coef)'].iloc[0])
    metricType.append('HR')

    mm.append(modality)
    surv.append('RFS-1y')
    modeling.append('continuous')
    unimulti.append('uni')
    metric.append( res.summary['p'].iloc[0])
    metricType.append('p-value')


    print('log')
    # univariate continuous, log scale
    cph = CoxPHFitter()
    res = cph.fit(dataME, duration_col='RFS_days2', event_col='Oneyear_RFS_event', formula=('T0_%s_TFE_log' % modality))
    res.print_summary()

    mm.append(modality)
    surv.append('RFS-1y')
    modeling.append('log')
    unimulti.append('uni')
    metric.append( res.summary['exp(coef)'].iloc[0])
    metricType.append('HR')

    mm.append(modality)
    surv.append('RFS-1y')
    modeling.append('log')
    unimulti.append('uni')
    metric.append( res.summary['p'].iloc[0])
    metricType.append('p-value')


    # multivariate binary
    print('multivariate binary')
    cph = CoxPHFitter()
    res = cph.fit(dataME_mv, duration_col='RFS_days2', event_col='Oneyear_RFS_event', formula=('ctDNAhigh_%s + rightsided_primary + rectal_primary + fongHigh + age_std + isMale + metachronous' % modality))
    res.print_summary()

    mm.append(modality)
    surv.append('RFS-1y')
    modeling.append('cutoff')
    unimulti.append('multi')
    metric.append( res.summary['exp(coef)'].iloc[0])
    metricType.append('HR')

    mm.append(modality)
    surv.append('RFS-1y')
    modeling.append('cutoff')
    unimulti.append('multi')
    metric.append( res.summary['p'].iloc[0])
    metricType.append('p-value')


    # multivariate continuous
    print('multivariate continuous, by 10')
    cph = CoxPHFitter()
    res = cph.fit(dataME_mv, duration_col='RFS_days2', event_col='Oneyear_RFS_event', formula=('T0_%s_TFE_10 + rightsided_primary + rectal_primary + fongHigh + age_std + isMale + metachronous' % modality))
    res.print_summary()

    mm.append(modality)
    surv.append('RFS-1y')
    modeling.append('continuous')
    unimulti.append('multi')
    metric.append( res.summary['exp(coef)'].iloc[0])
    metricType.append('HR')

    mm.append(modality)
    surv.append('RFS-1y')
    modeling.append('continuous')
    unimulti.append('multi')
    metric.append( res.summary['p'].iloc[0])
    metricType.append('p-value')


    # multivariate continuous
    print('multivariate log')
    cph = CoxPHFitter()
    res = cph.fit(dataME_mv, duration_col='RFS_days2', event_col='Oneyear_RFS_event', formula=('T0_%s_TFE_log + rightsided_primary + rectal_primary + fongHigh + age_std + isMale + metachronous' % modality))
    res.print_summary()

    mm.append(modality)
    surv.append('RFS-1y')
    modeling.append('log')
    unimulti.append('multi')
    metric.append( res.summary['exp(coef)'].iloc[0])
    metricType.append('HR')

    mm.append(modality)
    surv.append('RFS-1y')
    modeling.append('log')
    unimulti.append('multi')
    metric.append( res.summary['p'].iloc[0])
    metricType.append('p-value')

print('\n\n\n\n')
# # multi-organ competing risks
# from lifelines import AalenJohansenFitter
#
# event = np.zeros(dataME.shape[0])
#
# # those who are censored, are already 0
#
# # those who get liver-only recurrence are a competing risk
# rec = np.where(dataME['RFS_event'] == 1)[0]
#
# # missing
# missing = np.where(dataME['multiorgan_recurrence'].isna())[0]
#
# event[np.intersect1d(rec, missing)] = np.nan
#
# # one organ, competing
# liver = np.where(dataME['multiorgan_recurrence'] == 0)[0]
# event[np.intersect1d(rec, liver)] = 2
#
# # multi-organ
# mo = np.where(dataME['multiorgan_recurrence'] == 1)[0]
# event[np.intersect1d(rec, mo)] = 1
#
# finite = np.where(np.isfinite(event))[0]
#
# ajf = AalenJohansenFitter(calculate_variance=True)
# ajf.fit(dataME.iloc[finite]['RFS_days'], pd.Series(event[finite].astype(int), index=dataME.iloc[finite].index), event_of_interest=1)
# ajf.cumulative_density_
# ajf.plot()
#
# # censor 0, interest 1, competing 2
# dataME['event_cmprsk'] = event
#
# ###########################################################################
# # the same for 1-year, so everyone censored at 365 days
# event2 = np.zeros(dataME.shape[0])
# # those who get liver-only recurrence are a competing risk
# rec = np.where(dataME['Oneyear_RFS_event'] == 1)[0]
#
# # missing
# missing = np.where(dataME['multiorgan_recurrence'].isna())[0]
#
# event2[np.intersect1d(rec, missing)] = np.nan
#
# # one organ, competing
# liver = np.where(dataME['multiorgan_recurrence'] == 0)[0]
# event2[np.intersect1d(rec, liver)] = 2
#
# # multi-organ
# mo = np.where(dataME['multiorgan_recurrence'] == 1)[0]
# event2[np.intersect1d(rec, mo)] = 1
#
# dataME['event_cmprsk_1y'] = event2


############### OS
print('################################\nOS\n################################')

T = dataME['OS_days']
E = dataME['OS_event']

for modality in modalities:
    fig = plt.figure()
    ax =fig.add_subplot(111)

    dem = (dataME['ctDNAhigh_%s' % modality] == 1)

    kmfH = KaplanMeierFitter()
    kmfH.fit(T[dem], event_observed=E[dem], label='ctDNAhigh')
    kmfH.plot_survival_function(ax=ax, show_censors=True, color='C7', ci_show=True)

    print('ctDNA high, median survival %f days' % kmfH.median_survival_time_)

    kmfL = KaplanMeierFitter()
    kmfL.fit(T[~dem], event_observed=E[~dem], label='ctDNAlow')
    kmfL.plot_survival_function(ax=ax, show_censors=True, color='C8', ci_show=True)
    print('ctDNA low, median survival %f days' % kmfL.median_survival_time_)

    ax.set_xticks(365.25*np.arange(8))
    ax.set_xticklabels(np.arange(8))

    ax.set_ylim(0.0, 1.0)

    ax.set_xlabel('Time (years)')

    ax.set_ylabel('Overall survival')


    results = logrank_test(T[dem], T[~dem], E[dem], E[~dem])
    results.print_summary()

    add_at_risk_counts(kmfH, kmfL, ax=ax)
    plt.tight_layout()
    if args.saveplots:
        fig.savefig('../figures/T0_%s_OS.png' % modality, dpi=600)

    print('binary')
    cph = CoxPHFitter()
    res = cph.fit(dataME, duration_col='OS_days', event_col='OS_event', formula=('ctDNAhigh_%s' % modality))
    res.print_summary()

    print(res.summary.iloc[:,[1,5,6,-2]])


    mm.append(modality)
    surv.append('OS')
    modeling.append('cutoff')
    unimulti.append('uni')
    metric.append( res.summary['exp(coef)'].iloc[0])
    metricType.append('HR')

    mm.append(modality)
    surv.append('OS')
    modeling.append('cutoff')
    unimulti.append('uni')
    metric.append( res.summary['p'].iloc[0])
    metricType.append('p-value')



    print('\nOS\n')
    # univariate continuous
    print('continuous, by 10')
    cph = CoxPHFitter()
    res = cph.fit(dataME, duration_col='OS_days', event_col='OS_event', formula=('T0_%s_TFE_10' % modality))
    res.print_summary()

    print(res.summary.iloc[:,[1,5,6,-2]])

    mm.append(modality)
    surv.append('OS')
    modeling.append('continuous')
    unimulti.append('uni')
    metric.append( res.summary['exp(coef)'].iloc[0])
    metricType.append('HR')

    mm.append(modality)
    surv.append('OS')
    modeling.append('continuous')
    unimulti.append('uni')
    metric.append( res.summary['p'].iloc[0])
    metricType.append('p-value')


    print('\nlog')
    # univariate continuous, log scale
    cph = CoxPHFitter()
    res = cph.fit(dataME, duration_col='OS_days', event_col='OS_event', formula=('T0_%s_TFE_log' % modality))
    res.print_summary()

    mm.append(modality)
    surv.append('OS')
    modeling.append('log')
    unimulti.append('uni')
    metric.append( res.summary['exp(coef)'].iloc[0])
    metricType.append('HR')

    mm.append(modality)
    surv.append('OS')
    modeling.append('log')
    unimulti.append('uni')
    metric.append( res.summary['p'].iloc[0])
    metricType.append('p-value')


    # multivariate binary
    print('\n\nmultivariate binary')
    cph = CoxPHFitter()
    res = cph.fit(dataME_mv, duration_col='OS_days', event_col='OS_event', formula=('ctDNAhigh_%s + rightsided_primary + rectal_primary + fongHigh + age_std + isMale + metachronous' % modality))
    res.print_summary()

    mm.append(modality)
    surv.append('OS')
    modeling.append('cutoff')
    unimulti.append('multi')
    metric.append( res.summary['exp(coef)'].iloc[0])
    metricType.append('HR')

    mm.append(modality)
    surv.append('OS')
    modeling.append('cutoff')
    unimulti.append('multi')
    metric.append( res.summary['p'].iloc[0])
    metricType.append('p-value')



    # multivariate continuous
    print('\n\nmultivariate continuous, by 10')
    cph = CoxPHFitter()
    res = cph.fit(dataME_mv, duration_col='OS_days', event_col='OS_event', formula=('T0_%s_TFE_10 + rightsided_primary + rectal_primary + fongHigh + age_std + isMale + metachronous' % modality))
    res.print_summary()

    mm.append(modality)
    surv.append('OS')
    modeling.append('continuous')
    unimulti.append('multi')
    metric.append( res.summary['exp(coef)'].iloc[0])
    metricType.append('HR')

    mm.append(modality)
    surv.append('OS')
    modeling.append('continuous')
    unimulti.append('multi')
    metric.append( res.summary['p'].iloc[0])
    metricType.append('p-value')



    # multivariate continuous
    print('\nmultivariate log')
    cph = CoxPHFitter()
    res = cph.fit(dataME_mv, duration_col='OS_days', event_col='OS_event', formula=('T0_%s_TFE_log + rightsided_primary + rectal_primary + fongHigh + age_std + isMale + metachronous' % modality))
    res.print_summary()

    mm.append(modality)
    surv.append('OS')
    modeling.append('log')
    unimulti.append('multi')
    metric.append( res.summary['exp(coef)'].iloc[0])
    metricType.append('HR')

    mm.append(modality)
    surv.append('OS')
    modeling.append('log')
    unimulti.append('multi')
    metric.append( res.summary['p'].iloc[0])
    metricType.append('p-value')




results = pd.DataFrame({'modality':mm, 'outcome': surv, 'tfe_transform': modeling, 'analysis': unimulti, 'value':metric, 'measure': metricType})
for surv in ['RFS-1y', 'OS']:
    resultsTmp = results[results['outcome'] == surv]
    resultsTmp = resultsTmp[resultsTmp['measure'] == 'HR']
    for analysis in ['uni', 'multi']:
        resultsTmp = resultsTmp[resultsTmp['analysis'] == analysis]




sys.exit(0)
#############################################################################################################################
# classifiers
#############################################################################################################################
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, roc_curve, precision_recall_curve, average_precision_score, roc_auc_score
from copy import deepcopy

showtrain = False

labelNames = ['Oneyear_RFS_event', 'OS_binary', 'multiorgan_recurrence']
features = ['T0_medseq_TFE_log', 'rightsided_primary', 'rectal_primary', 'fongHigh', 'age_std', 'isMale', 'metachronous']

taskName = {'Oneyear_RFS_event': '1-year RFS', 'OS_binary': '3-year OS', 'multiorgan_recurrence': 'multiorgan recurrence'}

featureNames = {'T0_medseq_TFE_log': 'log(TFE)', 'rightsided_primary': 'location: right', 'rectal_primary': 'location: rectum', 'fongHigh': 'Fong score: high', 'age_std': 'age', 'isMale': 'sex: male', 'metachronous': 'metachronous', 'KRAS_mutant': 'KRAS mutant'}

metricNames = ['ROC AUC', 'PR AUC', 'F1 score', 'PPV', 'sensitivity', 'specificity']

for labelName in labelNames:
    print('\n#######################################\n%s\n#######################################' % labelName)

    X = deepcopy(dataME_mv)
    if labelName == 'Oneyear_RFS_event':
        # all good, nothing to edit
        pass

    elif labelName == 'OS_binary':
        # 3-years
        daysCutoff = 365.25 * 3

        # 0: alive after 3 years
        # 1: dead before 3 years
        # NaN: censored before 3 years
        y = np.zeros(X.shape[0])

        # dead before 3years
        ii = np.where(np.logical_and(X['OS_days'] < daysCutoff, X['OS_event'] == 1))[0]
        y[ii] = 1

        # censored before 3 years
        ii = np.where(np.logical_and(X['OS_days'] < daysCutoff, X['OS_event'] == 0))[0]
        y[ii] = np.nan


        X['OS_binary'] = y

        X = X[~X['OS_binary'].isna()]

    elif labelName == 'multiorgan_recurrence':
        X = X[~X['multiorgan_recurrence'].isna()]
        features.append('KRAS_mutant')

    else:
        raise ValueError('unknown task')

    print('total sample size: %d\n#######################################' % X.shape[0])


    # roc and pr
    fig, ax = plt.subplots(1,1)
    fig2, ax2 = plt.subplots(1,1)

    cv = StratifiedKFold(n_splits=args.nFolds, shuffle=True, random_state=42)

    allthresholds = np.array([])

    labelArchive = []
    posteriorArchive = []
    predictionArchive = []

    rocAucFold = np.zeros(args.nFolds)
    prAucFold = np.zeros(rocAucFold.shape)
    f1Fold = np.zeros(rocAucFold.shape)
    precisionFold = np.zeros(rocAucFold.shape)
    recallFold = np.zeros(rocAucFold.shape)
    specificityFold = np.zeros(rocAucFold.shape)

    rocAucFoldRnd = np.zeros((args.nFolds, args.nPermutations))
    prAucFoldRnd = np.zeros(rocAucFoldRnd.shape)

    for i, (trainInd, testInd) in enumerate(cv.split(X, X[labelName])):
        # print('\nfold %d' % i)
        Xtrain = X.iloc[trainInd][features]
        Xtest = X.iloc[testInd][features]
        ytrain = X.iloc[trainInd][labelName]
        ytest = X.iloc[testInd][labelName]

        labelArchive.append(ytest)

        clf = LogisticRegression(random_state=42)
        clf.fit(Xtrain, ytrain)

        if showtrain:
            print('Train')
            posteriors = clf.predict_proba(Xtrain)[:,1]
            predictions = clf.predict(Xtrain)
            print(precision_recall_fscore_support(ytrain, predictions, average='binary'))

            fpr, tpr, _ = roc_curve(ytrain, posteriors)
            pr, rc, _ = precision_recall_curve(ytrain, posteriors)


        posteriors = clf.predict_proba(Xtest)[:,1]
        predictions = clf.predict(Xtest)

        predictionArchive.append(predictions)
        posteriorArchive.append(posteriors)

        metricsFold = precision_recall_fscore_support(ytest, predictions, average=None)
        rocAucFold[i] = roc_auc_score(ytest, posteriors)
        prAucFold[i] = average_precision_score(ytest, posteriors)

        for j in range(args.nPermutations):
            ytestPermuted = np.random.permutation(ytest)
            rocAucFoldRnd[i,j] = roc_auc_score(ytestPermuted, posteriors)
            prAucFoldRnd[i,j] = average_precision_score(ytestPermuted, posteriors)

        precisionFold[i] = metricsFold[0][1]
        recallFold[i] = metricsFold[1][1]
        specificityFold[i] = metricsFold[1][0]
        f1Fold[i] = metricsFold[2][1]

        fpr, tpr, foldThresholdsROC = roc_curve(ytest, posteriors)
        # axes[1,0].plot(fpr, tpr, color=('C%d' % i), label=('fold %d' % i))
        allthresholds = np.union1d(allthresholds, foldThresholdsROC)

        pr, rc, foldThresholdsPR = precision_recall_curve(ytest, posteriors)
        # axes[1,1].plot(rc, pr, color=('C%d' % i), label=('fold %d' % i))
        allthresholds = np.union1d(allthresholds, foldThresholdsPR)


    allthresholds = np.sort(allthresholds)


    precision = np.zeros((args.nFolds, allthresholds.shape[0]))
    # also sensitivity
    recall = np.zeros(precision.shape)
    specificity = np.zeros(precision.shape)

    linestyles = ['-', ':', '-.']

    for i in range(args.nFolds):
        post = posteriorArchive[i]
        ytest = labelArchive[i]

        for j, t in enumerate(allthresholds):
            pred = (post >= t).astype(int)
            pr, rc, _, _ = precision_recall_fscore_support(ytest, pred, average=None)

            precision[i,j] = pr[1]
            recall[i,j] = rc[1]
            specificity[i,j] = rc[0]

        ax.plot(1-specificity[i], recall[i], color=('C9'), linestyle=linestyles[i], alpha=0.4, label=('fold %d' % i))
        ax2.plot(recall[i], precision[i], color=('C9'), linestyle=linestyles[i], alpha=0.4, label=('fold %d' % i) )

    muRecall = np.mean(recall,0)
    muPrecision = np.mean(precision,0)

    ax.plot(1-np.mean(specificity,0), muRecall, color='C0', label='mean')
    ax2.plot(muRecall, muPrecision, color='C0', label='mean')


    ax.set_title(r'%s, AUC = %.3f $\pm$ %.3f' % (taskName[labelName], np.mean(rocAucFold), np.std(rocAucFold, ddof=1)/np.sqrt(args.nFolds)), fontsize=11)
    ax2.set_title(r'%s, AUC = %.3f $\pm$ %.3f' % (taskName[labelName], np.mean(prAucFold), np.std(prAucFold, ddof=1)/np.sqrt(args.nFolds)), fontsize=11)

    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)

    ax2.set_ylabel('Positive Predictive Value/Precision', fontsize=13)
    ax2.set_xlabel('True Positive Rate/Recall', fontsize=13)

    xx = np.linspace(0,1,5)
    ax.plot(xx, xx, 'k--')

    yy = np.mean(X[labelName])
    ax2.axhline(yy, color='k', linestyle='--')

    lim = 0.05
    ax.set_xlim(0-lim,1+lim)
    ax.set_ylim(0-lim,1+lim)
    ax2.set_xlim(0-lim,1+lim)
    ax2.set_ylim(0-lim,1+lim)

    ax.legend()
    ax2.legend()

    plt.tight_layout()

    pvalueROC = np.mean(np.mean(rocAucFoldRnd,0) >= np.mean(rocAucFold))
    pvaluePR = np.mean(np.mean(prAucFoldRnd,0) >= np.mean(prAucFold))


    orderedMetrics = [rocAucFold, prAucFold, f1Fold, precisionFold, recallFold, specificityFold]
    for mm, mn in zip(orderedMetrics, metricNames):
        print('%s: %.3f +/- %.3f' % (mn, np.mean(mm), np.std(mm,ddof=1)/np.sqrt(args.nFolds)))
        if mn == 'ROC AUC':
            print('\tp-value = %.3f' % pvalueROC)
        if mn == 'PR AUC':
            print('\tp-value = %.3f' % pvaluePR)

    spc = np.mean(specificity, 0)
    delta = spc - 0.95
    delta[delta<0] = 10.
    index = np.argmin(delta)
    sigmaRecall = np.std(recall, axis=0, ddof=1) / np.sqrt(args.nFolds)

    print('Sensitivity at 95%% specificity: %.2f +/- %.3f' % (muRecall[index], sigmaRecall[index]))

    fig.savefig('../figures/prediction_%s_lr_roccurve.png' % labelName, dpi=600)
    fig2.savefig('../figures/prediction_%s_lr_prcurves.png' % labelName, dpi=600)

    #############################################################################################################################
    # bootstrapping
    np.random.seed(42)

    coefs = np.zeros((args.nBootstraps, len(features)))
    for i in range(args.nBootstraps):
        ind = np.random.choice(np.arange(X.shape[0]), size=X.shape[0], replace=True)
        Xtrain = X.iloc[ind][features]
        ytrain = X.iloc[ind][labelName]

        clf = LogisticRegression(random_state=42)
        clf.fit(Xtrain, ytrain)
        coefs[i] = clf.coef_

    fig, ax = plt.subplots(1,1)
    ax.boxplot(coefs, whis=(2.5, 97.5), vert=False, showfliers=False)
    ax.axvline(0, color='k', linestyle='--')

    ax.set_yticklabels([featureNames[f] for f in features])
    ax.set_xlabel('Logistic regression coefficient')
    ax.set_title('Feature importance')

    ax.set_title('%s' % taskName[labelName])

    plt.tight_layout()
    fig.savefig('../figures/prediction_%s_lr_feature_importance.png' % labelName, dpi=600)

    #############################################################################################################################
    # calibration
    ypost = np.hstack(posteriorArchive)
    y = np.hstack(labelArchive)

    ici = float(np.mean(np.abs(lowess(y, ypost, return_sorted=False) - ypost)))

    fig, ax = plt.subplots(1,1)
    CalibrationDisplay.from_predictions(y,ypost,pos_label=1,ax=ax)

    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title('Calibration curve %s, ICI=%.3f' % (taskName[labelName], ici))
    fig.savefig('../figures/prediction_%s_lr_calibration.png' % labelName, dpi=600)

    #############################################################################################################################

    # DCA
    rangeL = 1
    rangeH = 99

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # treat none
    ax.axhline(0.0, color='k', label='treat none')

    # calculate DC with step 0.01
    pp = np.arange(rangeL, rangeH+1) / 100
    oddratio = pp / (1-pp)

    # treat all
    n = y.shape[0]

    ypred = np.ones(n, int)
    tp = np.sum(np.logical_and(ypred==1, y==1)) / n
    fp = np.sum(np.logical_and(ypred==1,y==0)) / n
    fp *= oddratio

    nb = tp - fp

    ax.plot(pp, nb, color='C0', linestyle='--', label='treat all')


    # model
    nb = np.zeros(pp.shape[0])
    tp = np.zeros(nb.shape[0])
    fp = np.zeros(nb.shape[0])
    for i,thres in enumerate(pp):
        # go over all thresholds
        # if posterior > threshold, predict as positive
        ypred = (ypost >= thres).astype(int)
        # odds ratio
        oddsratio = thres / (1 - thres)

        tp[i] = np.sum(np.logical_and(ypred==1, y==1)) / n
        fp[i] = np.sum(np.logical_and(ypred==1, y==0)) / n

        nb[i] = tp[i] - (fp[i] * oddsratio)

    ax.plot(pp, nb, color='C5', linestyle='-', label='model')


    ax.set_xlabel('cut-off probability', fontsize=14)
    ax.set_ylabel('net benefit', fontsize=15)

    ax.set_ylim(-1,1)
    ax.set_xlim(0,0.6)

    ax.legend()
    ax.set_title('%s' % taskName[labelName])
    fig.savefig('../figures/prediction_%s_lr_dca.png' % labelName, dpi=600)


sys.exit(0)
#############################################################################################################################
#############################################################################################################################
import pymc as pm

# Extract data
features = ['T0_medseq_TFE_log', 'rightsided_primary', 'rectal_primary', 'fongHigh', 'age_std', 'isMale', 'metachronous']

X_cov = dataME_mv[features]
n_cov = X_cov.shape[1]

RFS_time = dataME_mv['RFS_days'].values
RFS_event = dataME_mv['RFS_event'].values
multi_met = dataME_mv['multiorgan_recurrence'].values

with pm.Model() as joint_model:

    # Hyperpriors for scale of effects
    sigma = pm.HalfNormal('sigma', sigma=1.0, shape=2)

    # LKJ prior for 2D correlation matrix
    packed_L, corr, stds = pm.LKJCholeskyCov('packed_L', n=2, eta=2.0, sd_dist=pm.HalfNormal.dist(1.0), compute_corr=True)
    # packed_L = pm.LKJCholeskyCov('packed_L', n=2, eta=2.0, sd_dist=pm.HalfNormal.dist(1.0))
    L = pm.expand_packed_triangular(2, packed_L, lower=True)

    # Matrix of effects: n_cov rows (features), 2 columns (outcomes)
    beta_raw = pm.Normal('beta_raw', mu=0, sigma=1, shape=(n_cov, 2))
    beta = pm.Deterministic('beta', pm.math.dot(beta_raw, packed_L.T) * sigma)

    # Intercepts
    alpha_surv = pm.Normal('alpha_surv', mu=0, sigma=2)
    alpha_logit = pm.Normal('alpha_logit', mu=0, sigma=2)

    # Linear predictors
    linpred_surv = alpha_surv + pm.math.dot(X_cov, beta[:, 0])
    linpred_logit = alpha_logit + pm.math.dot(X_cov, beta[:, 1])

    # Weibull PH parameterization
    k = pm.HalfNormal('k', sigma=2)  # shape
    lambda_ = pm.math.exp(linpred_surv)   # log-linear on hazard scale

    # Likelihood for survival (RFS)
    logp_event = pm.math.log(k) + (k - 1) * pm.math.log(RFS_time) + pm.math.log(lambda_) - (lambda_ * RFS_time**k)
    logp_cens = -(lambda_ * RFS_time**k)
    log_lik_surv = pm.math.switch(RFS_event, logp_event, logp_cens)
    pm.Potential('lik_surv', log_lik_surv.sum())

    # Likelihood for logistic model (multi-organ)
    # Only defined where RFS_event == 1 (i.e., we observed recurrence type)
    idx_observed = RFS_event == 1
    pm.Bernoulli('multi_met', logit_p=linpred_logit[idx_observed],
                 observed=multi_met[idx_observed])

with joint_model:
    prior_pred = pm.sample_prior_predictive(samples=1000)


#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
# classifier 3-year OS
#############################################################################################################################

# 3-years
daysCutoff = 365.25 * 3

# 0: alive after 3 years
# 1: dead before 3 years
# NaN: censored before 3 years

y = np.zeros(dataME_mv.shape[0])

# dead before 3years
ii = np.where(np.logical_and(dataME_mv['OS_days'] < daysCutoff, dataME_mv['OS_event'] == 1))[0]
y[ii] = 1

# censored before 3 years
ii = np.where(np.logical_and(dataME_mv['OS_days'] < daysCutoff, dataME_mv['OS_event'] == 0))[0]
y[ii] = np.nan

dataME_mv['OS_binary'] = y

Xos = dataME_mv[~dataME_mv['OS_binary'].isna()]

fig, axes = plt.subplots(2,2)

labelName = 'OS_binary'
features = ['T0_medseq_TFE_log', 'rightsided_primary', 'rectal_primary', 'fongHigh', 'age_std', 'isMale', 'metachronous']
cv = StratifiedKFold(n_splits=args.nFolds, shuffle=True, random_state=42)

allthresholds = np.array([])

labelArchive = []
posteriorArchive = []
predictionArchive = []


for i, (trainInd, testInd) in enumerate(cv.split(Xos, Xos[labelName])):
    print('\nfold %d' % i)
    Xtrain = Xos.iloc[trainInd][features]
    Xtest = Xos.iloc[testInd][features]
    ytrain = Xos.iloc[trainInd][labelName]
    ytest = Xos.iloc[testInd][labelName]

    labelArchive.append(ytest)

    # clf = LinearSVC(dual=True, random_state=42)
    clf = LogisticRegression(random_state=42)
    clf.fit(Xtrain, ytrain)

    print('Train')
    # posteriors = clf.decision_function(Xtrain)
    posteriors = clf.predict_proba(Xtrain)[:,1]
    predictions = clf.predict(Xtrain)
    print(precision_recall_fscore_support(ytrain, predictions, average='binary'))

    fpr, tpr, _ = roc_curve(ytrain, posteriors)
    axes[0,0].plot(fpr, tpr, color=('C%d' % i), label=('fold %d' % i))

    pr, rc, _ = precision_recall_curve(ytrain, posteriors)
    axes[0,1].plot(rc, pr, color=('C%d' % i), label=('fold %d' % i))

    print('\nTest')
    # posteriors = clf.decision_function(Xtest)
    posteriors = clf.predict_proba(Xtest)[:,1]
    predictions = clf.predict(Xtest)

    predictionArchive.append(predictions)
    posteriorArchive.append(posteriors)

    #print(precision_recall_fscore_support(ytest, predictions, average='binary'))
    # print(precision_recall_fscore_support(ytest, predictions, average=None))

    fpr, tpr, foldThresholdsROC = roc_curve(ytest, posteriors)
    axes[1,0].plot(fpr, tpr, color=('C%d' % i), label=('fold %d' % i))
    allthresholds = np.union1d(allthresholds, foldThresholdsROC)
    print(allthresholds.shape[0])
    pr, rc, foldThresholdsPR = precision_recall_curve(ytest, posteriors)
    axes[1,1].plot(rc, pr, color=('C%d' % i), label=('fold %d' % i))
    allthresholds = np.union1d(allthresholds, foldThresholdsPR)
    print(allthresholds.shape[0])


allthresholds = np.sort(allthresholds)

fig2, axes2 = plt.subplots(1,2)

precision = np.zeros((args.nFolds, allthresholds.shape[0]))
# also sensitivity
recall = np.zeros(precision.shape)
specificity = np.zeros(precision.shape)

linestyles = ['-', ':', '-.']

for i in range(args.nFolds):
    post = posteriorArchive[i]
    ytest = labelArchive[i]

    for j, t in enumerate(allthresholds):
        pred = (post >= t).astype(int)
        pr, rc, _, _ = precision_recall_fscore_support(ytest, pred, average=None)

        precision[i,j] = pr[1]
        recall[i,j] = rc[1]
        specificity[i,j] = rc[0]

    axes2[0].plot(1-specificity[i], recall[i], color=('C9'), linestyle=linestyles[i], alpha=0.4, label=('fold %d' % i))
    axes2[1].plot(recall[i], precision[i], color=('C9'), linestyle=linestyles[i], alpha=0.4, label=('fold %d' % i) )

muRecall = np.mean(recall,0)
muPrecision = np.mean(precision,0)

axes2[0].plot(1-np.mean(specificity,0), muRecall, color='C0', label='mean')
axes2[1].plot(muRecall, muPrecision, color='C0', label='mean')

# sigmaRecall = np.std(recall, axis=0, ddof=1)
# sigmaPrecision = np.std(precision, axis=0, ddof=1)

# axes2[0].fill_between(1-np.mean(specificity,0), muRecall-sigmaRecall, muRecall+sigmaRecall, color='C9', alpha=0.2)
# axes2[1].fill_between(muRecall, muPrecision-sigmaPrecision, muPrecision+sigmaPrecision, color='C9', alpha=0.2)




axes2[0].set_title('ROC curve', fontsize=14)
axes2[1].set_title('Precision-Recall curve', fontsize=14)

axes2[0].set_xlabel('False Positive Rate', fontsize=13)
axes2[0].set_ylabel('True Positive Rate', fontsize=13)

axes2[1].set_ylabel('Positive Predictive Value/Precision', fontsize=13)
axes2[1].set_xlabel('True Positive Rate/Recall', fontsize=13)



#
# axes[0,0].set_title('train roc')
# axes[0,1].set_title('train pr')
# axes[1,0].set_title('test roc')
# axes[1,1].set_title('test pr')

xx = np.linspace(0,1,5)
axes2[0].plot(xx, xx, 'k--')
# axes[1,0].plot(xx, xx, 'k--')


yy = np.mean(dataME_mv[labelName])
axes2[1].axhline(yy, color='k', linestyle='--')
# axes[1,1].axhline(yy, color='k', linestyle='--')

lim = 0.05
axes2[0].set_xlim(0-lim,1+lim)
axes2[0].set_ylim(0-lim,1+lim)
axes2[1].set_xlim(0-lim,1+lim)
axes2[1].set_ylim(0-lim,1+lim)
# axes[1,0].set_xlim(0-lim,1+lim)
# axes[1,0].set_ylim(0-lim,1+lim)
# axes[1,1].set_xlim(0-lim,1+lim)
# axes[1,1].set_ylim(0-lim,1+lim)


axes2[0].legend()
axes2[1].legend()
# axes[1,0].legend()
# axes[1,1].legend()
plt.tight_layout()
fig2.savefig('../figures/prediction_os36_lr_roccurves.png', dpi=600)


#############################################################################################################################
# calibration
ypost = np.hstack(posteriorArchive)
y = np.hstack(labelArchive)

fig, ax = plt.subplots(1,1)
CalibrationDisplay.from_predictions(y,ypost,pos_label=1,ax=ax)
fig.savefig('../figures/prediction_os36_lr_calibration.png', dpi=600)

#############################################################################################################################
# DCA

rangeL = 1
rangeH = 99

fig = plt.figure()
ax = fig.add_subplot(111)

# treat none
ax.axhline(0.0, color='k', label='treat none')

# calculate DC with step 0.01
pp = np.arange(rangeL, rangeH+1) / 100
oddratio = pp / (1-pp)

# treat all
n = y.shape[0]

ypred = np.ones(n, int)
tp = np.sum(np.logical_and(ypred==1, y==1)) / n
fp = np.sum(np.logical_and(ypred==1,y==0)) / n
fp *= oddratio

nb = tp - fp

ax.plot(pp, nb, color='C0', linestyle='--', label='treat all')


# model
nb = np.zeros(pp.shape[0])
tp = np.zeros(nb.shape[0])
fp = np.zeros(nb.shape[0])
for i,thres in enumerate(pp):
    # go over all thresholds
    # if posterior > threshold, predict as positive
    ypred = (ypost >= thres).astype(int)
    # odds ratio
    oddsratio = thres / (1 - thres)

    tp[i] = np.sum(np.logical_and(ypred==1, y==1)) / n
    fp[i] = np.sum(np.logical_and(ypred==1, y==0)) / n

    nb[i] = tp[i] - (fp[i] * oddsratio)

ax.plot(pp, nb, color='C5', linestyle='-', label='model')


ax.set_xlabel('cut-off probability', fontsize=14)
ax.set_ylabel('net benefit', fontsize=15)

ax.set_ylim(-1,1)
ax.set_xlim(0,0.6)

ax.legend()
fig.savefig('../figures/prediction_os36_lr_dca.png', dpi=600)

#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
# multi-organ recurrence
#############################################################################################################################

dataMOR = dataME_mv[~dataME_mv['multiorgan_recurrence'].isna()]


fig, axes = plt.subplots(2,2)

labelName = 'multiorgan_recurrence'
features = ['T0_medseq_TFE_log', 'rightsided_primary', 'rectal_primary', 'fongHigh', 'age_std', 'isMale', 'metachronous', 'KRAS_mutant']
cv = StratifiedKFold(n_splits=nFolds, shuffle=True, random_state=42)

allthresholds = np.array([])

labelArchive = []
posteriorArchive = []
predictionArchive = []


for i, (trainInd, testInd) in enumerate(cv.split(dataMOR, dataMOR[labelName])):
    print('\nfold %d' % i)
    Xtrain = dataMOR.iloc[trainInd][features]
    Xtest = dataMOR.iloc[testInd][features]
    ytrain = dataMOR.iloc[trainInd][labelName]
    ytest = dataMOR.iloc[testInd][labelName]

    labelArchive.append(ytest)

    # clf = LinearSVC(dual=True, random_state=42)
    clf = LogisticRegression(random_state=42)
    clf.fit(Xtrain, ytrain)

    print('Train')
    # posteriors = clf.decision_function(Xtrain)
    posteriors = clf.predict_proba(Xtrain)[:,1]
    predictions = clf.predict(Xtrain)
    print(precision_recall_fscore_support(ytrain, predictions, average='binary'))

    fpr, tpr, _ = roc_curve(ytrain, posteriors)
    axes[0,0].plot(fpr, tpr, color=('C%d' % i), label=('fold %d' % i))

    pr, rc, _ = precision_recall_curve(ytrain, posteriors)
    axes[0,1].plot(rc, pr, color=('C%d' % i), label=('fold %d' % i))

    print('\nTest')
    # posteriors = clf.decision_function(Xtest)
    posteriors = clf.predict_proba(Xtest)[:,1]
    predictions = clf.predict(Xtest)

    predictionArchive.append(predictions)
    posteriorArchive.append(posteriors)

    #print(precision_recall_fscore_support(ytest, predictions, average='binary'))
    # print(precision_recall_fscore_support(ytest, predictions, average=None))

    fpr, tpr, foldThresholdsROC = roc_curve(ytest, posteriors)
    axes[1,0].plot(fpr, tpr, color=('C%d' % i), label=('fold %d' % i))
    allthresholds = np.union1d(allthresholds, foldThresholdsROC)
    print(allthresholds.shape[0])
    pr, rc, foldThresholdsPR = precision_recall_curve(ytest, posteriors)
    axes[1,1].plot(rc, pr, color=('C%d' % i), label=('fold %d' % i))
    allthresholds = np.union1d(allthresholds, foldThresholdsPR)
    print(allthresholds.shape[0])


allthresholds = np.sort(allthresholds)

fig2, axes2 = plt.subplots(1,2)

precision = np.zeros((args.nFolds, allthresholds.shape[0]))
# also sensitivity
recall = np.zeros(precision.shape)
specificity = np.zeros(precision.shape)

linestyles = ['-', ':', '-.']

for i in range(args.nFolds):
    post = posteriorArchive[i]
    ytest = labelArchive[i]

    for j, t in enumerate(allthresholds):
        pred = (post >= t).astype(int)
        pr, rc, _, _ = precision_recall_fscore_support(ytest, pred, average=None)

        precision[i,j] = pr[1]
        recall[i,j] = rc[1]
        specificity[i,j] = rc[0]

    axes2[0].plot(1-specificity[i], recall[i], color=('C9'), linestyle=linestyles[i], alpha=0.4, label=('fold %d' % i))
    axes2[1].plot(recall[i], precision[i], color=('C9'), linestyle=linestyles[i], alpha=0.4, label=('fold %d' % i) )

muRecall = np.mean(recall,0)
muPrecision = np.mean(precision,0)

axes2[0].plot(1-np.mean(specificity,0), muRecall, color='C0', label='mean')
axes2[1].plot(muRecall, muPrecision, color='C0', label='mean')

# sigmaRecall = np.std(recall, axis=0, ddof=1)
# sigmaPrecision = np.std(precision, axis=0, ddof=1)

# axes2[0].fill_between(1-np.mean(specificity,0), muRecall-sigmaRecall, muRecall+sigmaRecall, color='C9', alpha=0.2)
# axes2[1].fill_between(muRecall, muPrecision-sigmaPrecision, muPrecision+sigmaPrecision, color='C9', alpha=0.2)




axes2[0].set_title('ROC curve', fontsize=14)
axes2[1].set_title('Precision-Recall curve', fontsize=14)

axes2[0].set_xlabel('False Positive Rate', fontsize=13)
axes2[0].set_ylabel('True Positive Rate', fontsize=13)

axes2[1].set_ylabel('Positive Predictive Value/Precision', fontsize=13)
axes2[1].set_xlabel('True Positive Rate/Recall', fontsize=13)



#
# axes[0,0].set_title('train roc')
# axes[0,1].set_title('train pr')
# axes[1,0].set_title('test roc')
# axes[1,1].set_title('test pr')

xx = np.linspace(0,1,5)
axes2[0].plot(xx, xx, 'k--')
# axes[1,0].plot(xx, xx, 'k--')


yy = np.mean(dataME_mv[labelName])
axes2[1].axhline(yy, color='k', linestyle='--')
# axes[1,1].axhline(yy, color='k', linestyle='--')

lim = 0.05
axes2[0].set_xlim(0-lim,1+lim)
axes2[0].set_ylim(0-lim,1+lim)
axes2[1].set_xlim(0-lim,1+lim)
axes2[1].set_ylim(0-lim,1+lim)
# axes[1,0].set_xlim(0-lim,1+lim)
# axes[1,0].set_ylim(0-lim,1+lim)
# axes[1,1].set_xlim(0-lim,1+lim)
# axes[1,1].set_ylim(0-lim,1+lim)


axes2[0].legend()
axes2[1].legend()
# axes[1,0].legend()
# axes[1,1].legend()
plt.tight_layout()
fig2.savefig('../figures/prediction_mor_lr_roccurves.png', dpi=600)


#############################################################################################################################
# calibration
ypost = np.hstack(posteriorArchive)
y = np.hstack(labelArchive)

fig, ax = plt.subplots(1,1)
CalibrationDisplay.from_predictions(y,ypost,pos_label=1,ax=ax)
fig.savefig('../figures/prediction_mor_lr_calibration.png', dpi=600)

#############################################################################################################################
# DCA

rangeL = 1
rangeH = 99

fig = plt.figure()
ax = fig.add_subplot(111)

# treat none
ax.axhline(0.0, color='k', label='treat none')

# calculate DC with step 0.01
pp = np.arange(rangeL, rangeH+1) / 100
oddratio = pp / (1-pp)

# treat all
n = y.shape[0]

ypred = np.ones(n, int)
tp = np.sum(np.logical_and(ypred==1, y==1)) / n
fp = np.sum(np.logical_and(ypred==1,y==0)) / n
fp *= oddratio

nb = tp - fp

ax.plot(pp, nb, color='C0', linestyle='--', label='treat all')


# model
nb = np.zeros(pp.shape[0])
tp = np.zeros(nb.shape[0])
fp = np.zeros(nb.shape[0])
for i,thres in enumerate(pp):
    # go over all thresholds
    # if posterior > threshold, predict as positive
    ypred = (ypost >= thres).astype(int)
    # odds ratio
    oddsratio = thres / (1 - thres)

    tp[i] = np.sum(np.logical_and(ypred==1, y==1)) / n
    fp[i] = np.sum(np.logical_and(ypred==1, y==0)) / n

    nb[i] = tp[i] - (fp[i] * oddsratio)

ax.plot(pp, nb, color='C5', linestyle='-', label='model')


ax.set_xlabel('cut-off probability', fontsize=14)
ax.set_ylabel('net benefit', fontsize=15)

ax.set_ylim(-1,1)
ax.set_xlim(0,0.6)

ax.legend()
fig.savefig('../figures/prediction_os36_lr_dca.png', dpi=600)



#############################################################################################################################
# check after this point
#############################################################################################################################
# continuous
dataMEOS = dataMEOS[~dataMEOS['fongHigh'].isna()]
dataMEOS = dataMEOS[~dataMEOS['HGP_fully_desmoplastic'].isna()]

# dataMEOS['T0_medseq_TFE_log'] = np.log(dataMEOS['T0_medseq_TFE'])

cphC = CoxPHFitter()
resC = cphC.fit(dataMEOS, duration_col='OS_days', event_col='OS_event', formula='fongHigh + HGP_fully_desmoplastic + age + isMale + isR1')
resC.print_summary()


cph = CoxPHFitter()
res = cph.fit(dataMEOS, duration_col='OS_days', event_col='OS_event', formula='T0_medseq_TFE_log + fongHigh + HGP_fully_desmoplastic + age')
res.print_summary()




###############################################################
# oncomine negatives
dataMEonconeg = dataME[dataME['T0_vaf_TFE'] == 0.]

for i, (Tonconeg,Eonconeg) in enumerate(zip([dataMEonconeg['RFS_days'], dataMEonconeg['RFS_days2']], [dataMEonconeg['RFS_event'], dataMEonconeg['Oneyear_RFS_event']])):
    fig = plt.figure()
    ax =fig.add_subplot(111)

    dem = (dataMEonconeg['ctDNAhigh'] == 1)

    kmfH = KaplanMeierFitter()
    kmfH.fit(Tonconeg[dem], event_observed=Eonconeg[dem], label='MUT-/MEhigh')
    kmfH.plot_survival_function(ax=ax, show_censors=True, ci_show=True, color='C7')
    print('Oncomine-/medseq high, median survival %.1f days' % kmfH.median_survival_time_)

    kmfL = KaplanMeierFitter()
    kmfL.fit(Tonconeg[~dem], event_observed=Eonconeg[~dem], label='MUT-/MElow')
    kmfL.plot_survival_function(ax=ax, show_censors=True, ci_show=True, color='C8')
    print('Oncomine-/medseq low, median survival %.1f days' % kmfL.median_survival_time_)

    if i == 0:
        ax.set_xticks(30*np.array([0, 4, 8, 12, 16,20,24,28,32, 36]))
        ax.set_xticklabels(np.array([0, 4, 8, 12, 16,20,24,28,32, 36]))
        ax.set_xlim(0, 365*3 + 15)

        ax.set_xlabel('Time (months)')
        ax.axvline(360.0, color='k', linestyle='--')

    else:
        ax.set_xticks(np.arange(0,13,2) * 30)
        ax.set_xticklabels(np.arange(0,13,2))
        ax.set_xlabel('Time (months)')

    ax.set_ylabel('Recurrence-free survival')

    add_at_risk_counts(kmfH, kmfL, ax=ax)
    plt.tight_layout()

    results = logrank_test(Tonconeg[dem], Tonconeg[~dem], Eonconeg[dem], Eonconeg[~dem])
    results.print_summary()

    fig.savefig('../figures/KM_cfDNAhighlow_T0_oncomine_negatives_%d.png' % i, dpi=600)

# oncomine+, oncomine-/medseq+, oncomine-/medseq-
dataoncopos = data[data['T0_vaf_TFE'] > 0.]

dem2 = (dataoncopos['T0_vaf_TFE'] > 0.06)

Toncopos = dataoncopos['RFS_days']
Eoncopos = dataoncopos['RFS_event']


######## this is wrong
print('WARNING!')
# mixing RFS with RFS2
fig = plt.figure()
ax =fig.add_subplot(111)


kmf = KaplanMeierFitter()
kmf.fit(Tonconeg[dem], event_observed=Eonconeg[dem], label='oncomine-/ctDNAhigh')
kmf.plot_survival_function(ax=ax)

kmf = KaplanMeierFitter()
kmf.fit(Tonconeg[~dem], event_observed=Eonconeg[~dem], label='oncomine-/ctDNAlow')
kmf.plot_survival_function(ax=ax)

kmf = KaplanMeierFitter()
kmf.fit(Toncopos[dem2], event_observed=Eoncopos[dem2], label='oncomine+ high')
kmf.plot_survival_function(ax=ax)

kmf = KaplanMeierFitter()
kmf.fit(Toncopos[~dem2], event_observed=Eoncopos[~dem2], label='oncomine+ low')
kmf.plot_survival_function(ax=ax)


ax.set_xticks(365 * np.arange(6))
ax.set_xticklabels(np.arange(6))
ax.set_xlim(0, T.max()+10)

ax.set_xlabel('time (years)')

results = logrank_test(Tonconeg[dem], Tonconeg[~dem], Eonconeg[dem], Eonconeg[~dem])
results.print_summary()

fig.savefig('../figures/KM_cfDNAhighlow_T0_oncomine_negatives_incl_oncomine_pos.png', dpi=600)




sys.exit(0)

dataME['T0_medseq_TFE_log'] = np.log(dataME['T0_medseq_TFE'])

dataME = dataME[~dataME['HGP_fully_desmoplastic'].isna()]

# cox models

# univariate continuous
cph = CoxPHFitter()
res = cph.fit(dataME, duration_col='RFS_days2', event_col='OnRFS_event', formula='T0_medseq_TFE')
res.print_summary()

# univariate continuous, log scale
cph = CoxPHFitter()
res = cph.fit(dataME, duration_col='RFS_days', event_col='RFS_event', formula='T0_medseq_TFE_log')
res.print_summary()






# plot hazard ratio as function of tumor fraction

c = res.summary.iloc[0]['coef']
lb = res.summary.iloc[0]['coef lower 95%']
ub = res.summary.iloc[0]['coef upper 95%']

# tfe in %
x = np.arange(1,100)
hr = np.zeros(x.shape)
lhr = np.zeros(x.shape)
uhr = np.zeros(x.shape)

for i, tf in enumerate(x):
    s = np.log(tf / 100)
    e = np.log((1+tf) / 100)

    hr[i] = np.exp(c * e) / np.exp(c * s)
    lhr[i] = np.exp(lb * e) / np.exp(lb * s)
    uhr[i] = np.exp(ub * e) / np.exp(ub * s)


fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(x, hr, color='C0')
ax.fill_between(x, hr, lhr, color='C0', alpha=0.3)
ax.fill_between(x, uhr, hr, color='C0', alpha=0.3)

ax.axhline(1.0, color='k', linestyle='--')

ax.set_xlabel('Tumor fraction estimate', fontsize=16)
ax.set_ylabel('Hazard ratio', fontsize=16)

fig.savefig('../figures/hr_log_tfe_at_t0.png', dpi=600)

# let's add Fong
ii = np.where(np.isfinite(dataME['fongHigh']))[0]

cph = CoxPHFitter()
res = cph.fit(dataME.iloc[ii], duration_col='RFS_days', event_col='RFS_event', formula='fongHigh + T0_medseq_TFE_log')
res.print_summary()

cph2 = CoxPHFitter()
res2 = cph2.fit(dataME.iloc[ii], duration_col='RFS_days', event_col='RFS_event', formula='fongHigh')
res2.print_summary()


# among the Fong low
cph = CoxPHFitter()
res = cph.fit(dataME[dataME['fongHigh'] == 0], duration_col='RFS_days', event_col='RFS_event', formula='T0_medseq_TFE_log')
res.print_summary()
# nope

# actual end point, 1-year rfs
modelFong = logit('Oneyear_RFS_event ~ fongHigh', dataME.iloc[ii])
model = logit('Oneyear_RFS_event ~ fongHigh + T0_medseq_TFE_log', dataME.iloc[ii])

resFong = modelFong.fit()
res = model.fit()

print(resFong.summary())
print('----------------------------------------------------------------------------------------')
print(res.summary())

print('----------------------------------------------------------------------------------------')

# now with Cox
ii = np.where(np.isfinite(dataME['fongHigh']))[0]
cph = CoxPHFitter()
res = cph.fit(dataME.iloc[ii], duration_col='RFS_days2', event_col='Oneyear_RFS_event', formula='fongHigh + ctDNAhigh + age + isMale + rectal_primary + rightsided_primary')
res.print_summary()

cph = CoxPHFitter()
res = cph.fit(dataME.iloc[ii], duration_col='RFS_days2', event_col='Oneyear_RFS_event', formula='fongHigh + T0_medseq_TFE_log + age + isMale + rectal_primary + rightsided_primary')
res.print_summary()

######################
# simple train/test split, with bootstrap on the test set
from sklearn.model_selection import StratifiedShuffleSplit
ds = dataME.iloc[ii]

splitter = StratifiedShuffleSplit(n_splits=1, train_size=0.5, random_state=41)
for trn, tst in splitter.split(ds, ds['Oneyear_RFS_event']):
    dsTrain = ds.iloc[trn]
    dsTest = ds.iloc[tst]

    cphME = CoxPHFitter()
    res = cphME.fit(dsTrain, duration_col='RFS_days2', event_col='Oneyear_RFS_event', formula='fongHigh + T0_medseq_TFE_log + age + isMale + rectal_primary + rightsided_primary')

    cphC = CoxPHFitter()
    res = cphC.fit(dsTrain, duration_col='RFS_days2', event_col='Oneyear_RFS_event', formula='fongHigh + age + isMale + rectal_primary + rightsided_primary')

    phC = cphC.predict_partial_hazard(dsTest)
    phME = cphME.predict_partial_hazard(dsTest)

fig = plt.figure()
ax = fig.add_subplot(1,2,1)

xx = np.linspace(0,1,5)

ax.plot(xx, xx, 'k--')

[fpr, tpr, _] = roc_curve(dsTest['Oneyear_RFS_event'], phC)
ax.plot(fpr, tpr, color='C1', label='clinical')

[fpr, tpr, _] = roc_curve(dsTest['Oneyear_RFS_event'], phME)
ax.plot(fpr, tpr, color='C0', label='ME')

ax.legend()
#-----------------------------------------------------------------------------
ax = fig.add_subplot(1,2,2)

[pr, rc, _] = precision_recall_curve(dsTest['Oneyear_RFS_event'], phC)
ax.plot(rc, pr, color='C1', label='clinical')

[pr, rc, _] = precision_recall_curve(dsTest['Oneyear_RFS_event'], phME)
ax.plot(rc, pr, color='C0', label='ME')

ax.set_xlabel('Recall/TPR')
ax.set_ylabel('Precision/PPV')


ax.legend()


#######################
dataMUT = data[data['T0_VAF'] > 0.]
dataMUT['T3_ctDNApos'] = (dataMUT['T3_VAF'] > 0.002).astype(int)

ds = dataMUT[~dataMUT['fongHigh'].isna()]



splitter = StratifiedShuffleSplit(n_splits=1, train_size=0.5, random_state=41)
for trn, tst in splitter.split(ds, ds['Oneyear_RFS_event']):
    dsTrain = ds.iloc[trn]
    dsTest = ds.iloc[tst]

    cphMUT = CoxPHFitter()
    resMUT = cphMUT.fit(dsTrain, duration_col='RFS_days2', event_col='Oneyear_RFS_event', formula='fongHigh + T3_ctDNApos + age + isMale + rectal_primary + rightsided_primary')

    cphC = CoxPHFitter()
    resC = cphC.fit(dsTrain, duration_col='RFS_days2', event_col='Oneyear_RFS_event', formula='fongHigh + age + isMale + rectal_primary + rightsided_primary')

    phC = cphC.predict_partial_hazard(dsTest)
    phMUT = cphMUT.predict_partial_hazard(dsTest)


fig = plt.figure()
ax = fig.add_subplot(1,1,1)

xx = np.linspace(0,1,5)

ax.plot(xx, xx, 'k--')

[fpr, tpr, _] = roc_curve(dsTest['Oneyear_RFS_event'], phC)
ax.plot(fpr, tpr, color='C1', label='clinical')

[fpr, tpr, _] = roc_curve(dsTest['Oneyear_RFS_event'], phMUT)
ax.plot(fpr, tpr, color='C0', label='ME')

ax.legend()
#-----------------------------------------------------------------------------
# ax = fig.add_subplot(1,2,2)
#
# [pr, rc, _] = precision_recall_curve(dsTest['Oneyear_RFS_event'], phC)
# ax.plot(rc, pr, color='C1', label='clinical')
#
# [pr, rc, _] = precision_recall_curve(dsTest['Oneyear_RFS_event'], phMUT)
# ax.plot(rc, pr, color='C0', label='MUT')
#
# ax.set_xlabel('Recall/TPR')
# ax.set_ylabel('Precision/PPV')
#
#
# ax.legend()







######################

cv = StratifiedKFold(n_splits=Nfolds, shuffle=True, random_state=42)

ds = dataME[['Oneyear_RFS_event', 'fongHigh', 'isMale',  'age', 'T0_medseq_TFE_log']].dropna(axis=0)

Xclin = ds.iloc[:,1:-1]
Xme = ds.iloc[:,1:]

y = np.array(ds.iloc[:,0])

acc = np.zeros((2, Nfolds))
spc = np.zeros(acc.shape)
sens = np.zeros(acc.shape)
pr = np.zeros(acc.shape)
npv = np.zeros(acc.shape)
rocauc = np.zeros(acc.shape)
ici = np.zeros(acc.shape)
posteriors = np.zeros((2, ds.shape[0]))

for i, X in enumerate([Xclin, Xme]):
    for fold, (trainInd, testInd) in enumerate(cv.split(X,y)):
        XTrain = X.iloc[trainInd]
        XTest = X.iloc[testInd]
        ytrain = y[trainInd]
        ytest = y[testInd]

        model = LogisticRegression(C=1.0, random_state=42)
        model.fit(XTrain, ytrain)

        ypredClin = model.predict(XTest)
        ypostClin = model.predict_proba(XTest)[:,1]
        posteriors[i, testInd] = ypostClin

        acc[i, fold] = np.mean(ypredClin == ytest)
        prfold, rcfold, _, _ = precision_recall_fscore_support(ytest, ypredClin, average=None)

        npv[i,fold] = prfold[0]
        pr[i,fold] = prfold[1]

        spc[i,fold] = rcfold[0]
        sens[i,fold] = rcfold[1]

        rocauc[i,fold] = roc_auc_score(ytest, ypostClin)


        ici[i, fold] = float(np.mean(np.abs(lowess(ytest, ypostClin, return_sorted=False) - ypostClin)))



trueC, predC = calibration_curve(y, posteriors[0], pos_label=1, n_bins=8, strategy='uniform')
trueM, predM = calibration_curve(y, posteriors[1], pos_label=1, n_bins=8, strategy='uniform')

fig = plt.figure()
ax = fig.add_subplot(111)
disp = CalibrationDisplay.from_predictions(y, posteriors[0], n_bins=10, name='clin', ax=ax, color='C0')
disp = CalibrationDisplay.from_predictions(y, posteriors[1], n_bins=10, name='me', ax=ax, color='C1')


# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# ax.plot(trueC, predC, color='C0', label='clin')
# ax.plot(trueM, predM, color='C1', label='me')
# xx = np.linspace(0,1,5)
# ax.plot(xx, xx, 'k--')
#
# ax.legend()
how()
