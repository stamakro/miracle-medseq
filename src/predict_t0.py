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
parser.add_argument('--cnv-file', dest='cnvfile', metavar='CNVTFE', help='CNV tfe file', default='../data/medseqcnv_tumor_fraction_estimates_n120.csv', type=str)
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

modalities = ['medseq', 'cnv']

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


print('\n\n\n\n')


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



results = pd.DataFrame({'modality':mm, 'outcome': surv, 'tfe_transform': modeling, 'analysis': unimulti, 'value':metric, 'measure': metricType})
for surv in ['RFS-1y', 'OS']:
    resultsTmp = results[results['outcome'] == surv]
    resultsTmp = resultsTmp[resultsTmp['measure'] == 'HR']
    for analysis in ['uni', 'multi']:
        resultsTmp = resultsTmp[resultsTmp['analysis'] == analysis]
