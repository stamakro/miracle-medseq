import pandas as pd
import numpy as np
from rutils import *
import argparse
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
import pickle


def getChromosome(reg: str) -> str:
    return reg.split(':')[0]

parser = argparse.ArgumentParser(prog='deconvolve_standard.py', description='')

parser.add_argument('--clindb', dest='clindb', metavar='CLINICALDB', help='path to clean DB file', default='../data/cleanDB.csv')
parser.add_argument('--hbdfile', dest='hbdfile', metavar='HBDFILE', help='path to HBD methylation counts and sample info', default='../data/hbd_cpgi.csv')
parser.add_argument('--tumorfile', dest='tumorfile', metavar='TUMORFILE', help='path to tumor methylation counts and sample info', default='../data/crlm_cpgi.csv')
parser.add_argument('--cfdnafile', dest='miraclefile', metavar='CFDNAFILE', help='path to methylation counts of miracle samples', default='/home/stavros/emc/projects/MedSeq/processed/miracle-latest-2024-06-23-11-25/counts_aggregated_cpgi.csv')
parser.add_argument('--miraclecfdnainfo', dest='medseq_sample_info', metavar='INFOMEDSEQ', help='path to csv file containing CpG reads, mapping etc for MIRACLE cfdna samples', default='/home/stavros/Desktop/code/nb-deconv/medseq-deconv/data/new_sampleoverzicht_SW_20240219.csv')
parser.add_argument('--dmrfile', dest='dmr_file', metavar='DMRFILE', help='path to csv file containing DMRs, fold changes, etc', default='../results/dmrs_halfvariable_fwer_fc0_crlm_hbd_vafcor_in_exclusions.csv')
parser.add_argument('--tfedict', dest='tfe_dict', metavar='TFEFILE', help='path to pkl file containing tumor fraction estimates per lcode', default='../results/lcode2tfe.pkl')
parser.add_argument('--output', dest='outfile', metavar='OUTFILE', help='path to csv with entire dataset', default='../data/dataset.csv')

args = parser.parse_args()

dataHbd = pd.read_csv(args.hbdfile, index_col=0)
dataTumor = pd.read_csv(args.tumorfile, index_col=0)

countsHbd = dataHbd.iloc[np.where(pd.Series(dataHbd.index).apply(isAutosomal))[0]].astype(float)
countsTumor = dataTumor.iloc[np.where(pd.Series(dataTumor.index).apply(isAutosomal))[0]].astype(float)


librarysizeHbd = np.round(np.array(dataHbd.loc['Used reads'].astype(float)))
librarysizeTumor = np.round(np.array(dataTumor.loc['Used reads'].astype(float)))

dmrInfo = pd.read_csv(args.dmr_file, index_col=0)


dmrInfo = dmrInfo[dmrInfo['FDR_correlation_with_VAF_exclusion'] < 0.05]
dmrInfo = dmrInfo[dmrInfo['logFC_correlation_sign_matches'] == 1]


# 120 patients with successful baseline medseq
clinical = pd.read_csv(args.clindb, index_col=0)
clinical = clinical[clinical['Exclusion'] == 0.0]
clinical = clinical[clinical['T0_medseq_success'] == 1.0]

# also get the 11 T0 samples from excluded patients that had an oncomine hit
clinicalE = pd.read_csv(args.clindb, index_col=0)
clinicalE = clinicalE[clinicalE['Exclusion'] == 1.0]
clinicalE = clinicalE[clinicalE['T0_medseq_success'] == 1.0]
clinicalE = clinicalE[clinicalE['T0_VAF_oncomine'] > 0.0]
clinicalE = clinicalE[clinicalE['Reason exclusion'] != 'PA: levermetastasen van slokdarmCa']

miracleData = pd.read_csv(args.miraclefile, index_col=0)

# 8/120 do not have T3 sample after QC
miracleData0 = miracleData[clinical['T0_lcode']]
miracleData3 = miracleData[clinical[clinical['T3_medseq_success']==1]['T3_lcode']]
miracleDataE = miracleData[clinicalE['T0_lcode']]

miracleinfo = pd.read_csv(args.medseq_sample_info, index_col=0)
miracleinfo0 = miracleinfo.loc[clinical['T0_lcode']]
miracleinfo3 = miracleinfo.loc[clinical[clinical['T3_medseq_success']==1]['T3_lcode']]
miracleinfoE = miracleinfo.loc[clinicalE['T0_lcode']]

assert (miracleinfo0['Timepoint'] == 0).all()
assert (miracleinfo3['Timepoint'] == 3).all()

# reference atlas
countsBloodTumor = pd.concat((countsHbd, countsTumor), axis=1)
bloodtumormarkers = countsBloodTumor.loc[dmrInfo.index]

labels = np.ones(countsBloodTumor.shape[1], int)
labels[:countsHbd.shape[1]] = 0

# means and variances/dispersions
params = r_estimateMeanVariance(bloodtumormarkers.T, labels)
sys.exit(0)

lcode2tfe = dict()

for i, ds in enumerate([miracleDataE, miracleData0, miracleData3], 1):
    dataMarkers = ds.loc[bloodtumormarkers.index]

    tfe = r_estimateTF(dataMarkers.T, params['mus'], params['disps'])

    for lcode, tt in zip(dataMarkers.columns, tfe):
        assert lcode not in lcode2tfe
        lcode2tfe[lcode] = tt

    # break
sys.exit(0)
with open(args.tfe_dict, 'wb') as f:
    pickle.dump(lcode2tfe, f)

# evaluation with VAF, exclusions (oncomine)
lcode2vafE = clinicalE.set_index('T0_lcode')['T0_VAF_oncomine'].to_dict()

ll = list(clinicalE['T0_lcode'])

vaf0Eonc = np.zeros(len(lcode2vafE))
tfe0E = np.zeros(len(lcode2vafE))
for i, l in enumerate(ll):
    vaf0Eonc[i] = lcode2vafE[l]
    tfe0E[i] = lcode2tfe[l]


fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(vaf0Eonc, tfe0E)

M = np.maximum(np.max(vaf0Eonc), np.max(tfe0E))
xx = np.linspace(0, M, 4)
ax.plot(xx, xx, color='k', linestyle='--')

ax.set_xlabel('Oncomine VAF', fontsize=15)
ax.set_ylabel('TFE', fontsize=18)

ax.set_xlim(-0.01, M+0.01)
ax.set_ylim(-0.01, M+0.01)

rhoS, pS = spearmanr(vaf0Eonc, tfe0E)
rhoP, pP = pearsonr(vaf0Eonc, tfe0E)

print('exclusions (N=%d):' % len(tfe0E))
print('Pearson: %.2f\t%.5f' % (rhoP, pP))
print('Spearman: %.2f\t%.5f' % (rhoS, pS))
print('\n\n')

ax.set_title('N=%d, rho = %.2f' % (len(tfe0E), rhoS))


##### copy paste for T0
# evaluate for both Oncomine and qPCR

clinical['T0_medseq_TFE'] = clinical['T0_lcode'].map(lcode2tfe)
clinical['T3_medseq_TFE'] = clinical['T3_lcode'].map(lcode2tfe)



clinicalVAFpositive = clinical[clinical['T0_VAF_oncomine'] > 0]

fig = plt.figure()
ax = fig.add_subplot(1,2,1)

ax.scatter(clinicalVAFpositive['T0_VAF_oncomine'], clinicalVAFpositive['T0_medseq_TFE'])

M = np.maximum(np.max(clinicalVAFpositive['T0_VAF_oncomine']), np.max(clinicalVAFpositive['T0_medseq_TFE']))
xx = np.linspace(0, M, 4)
ax.plot(xx, xx, color='k', linestyle='--')

ax.set_xlabel('Oncomine VAF', fontsize=15)
ax.set_ylabel('TFE', fontsize=18)

ax.set_xlim(-0.01, M+0.01)
ax.set_ylim(-0.01, M+0.01)

rhoS, pS = spearmanr(clinicalVAFpositive['T0_VAF_oncomine'], clinicalVAFpositive['T0_medseq_TFE'])
rhoP, pP = pearsonr(clinicalVAFpositive['T0_VAF_oncomine'], clinicalVAFpositive['T0_medseq_TFE'])

print('T0, oncomine (N=%d):' % len(clinicalVAFpositive))
print('Pearson: %.2f\t%.5f' % (rhoP, pP))
print('Spearman: %.2f\t%.5f' % (rhoS, pS))
print('\n\n')

ax.set_title('oncomine: N=%d, rho = %.2f' % (len(clinicalVAFpositive), rhoS))

# pcr
ax = fig.add_subplot(1,2,2)

ax.scatter(clinicalVAFpositive['T0_VAF'], clinicalVAFpositive['T0_medseq_TFE'])

M = np.maximum(np.nanmax(clinicalVAFpositive['T0_VAF']), np.max(clinicalVAFpositive['T0_medseq_TFE']))
xx = np.linspace(0, M, 4)
ax.plot(xx, xx, color='k', linestyle='--')

ax.set_xlabel('qPCR VAF', fontsize=15)
ax.set_ylabel('TFE', fontsize=18)

ax.set_xlim(-0.01, M+0.01)
ax.set_ylim(-0.01, M+0.01)


finiteInd = np.where(~clinicalVAFpositive['T0_VAF'].isna())[0]
rhoS, pS = spearmanr(clinicalVAFpositive.iloc[finiteInd]['T0_VAF'], clinicalVAFpositive.iloc[finiteInd]['T0_medseq_TFE'])
rhoP, pP = pearsonr(clinicalVAFpositive.iloc[finiteInd]['T0_VAF'], clinicalVAFpositive.iloc[finiteInd]['T0_medseq_TFE'])

print('T0, dPCR (N=%d):' % len(finiteInd))
print('Pearson: %.2f\t%.5f' % (rhoP, pP))
print('Spearman: %.2f\t%.5f' % (rhoS, pS))
print('\n\n')

ax.set_title('dPCR: N=%d, rho = %.2f' % (len(finiteInd), rhoS))

############################
# T3

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.scatter(clinicalVAFpositive['T3_VAF'], clinicalVAFpositive['T3_medseq_TFE'])

M = np.maximum(np.nanmax(clinicalVAFpositive['T3_VAF']), np.max(clinicalVAFpositive['T3_medseq_TFE']))
xx = np.linspace(0, M, 4)
ax.plot(xx, xx, color='k', linestyle='--')

ax.set_xlabel('qPCR VAF', fontsize=15)
ax.set_ylabel('TFE', fontsize=18)

ax.set_xlim(-0.01, M+0.01)
ax.set_ylim(-0.01, M+0.01)


finiteInd = np.where(~clinicalVAFpositive['T3_VAF'].isna())[0]
rhoS, pS = spearmanr(clinicalVAFpositive.iloc[finiteInd]['T3_VAF'], clinicalVAFpositive.iloc[finiteInd]['T0_medseq_TFE'])
rhoP, pP = pearsonr(clinicalVAFpositive.iloc[finiteInd]['T3_VAF'], clinicalVAFpositive.iloc[finiteInd]['T0_medseq_TFE'])

print('T3, dPCR (N=%d):' % len(finiteInd))
print('Pearson: %.2f\t%.5f' % (rhoP, pP))
print('Spearman: %.2f\t%.5f' % (rhoS, pS))
print('\n\n')

ax.set_title('dPCR: N=%d, rho = %.2f' % (len(finiteInd), rhoS))


# make a new dataframe with the patients that are not excluded and have VAF or medseq tfe or CTC

clinical = pd.read_csv(args.clindb, index_col=0)
clinical = clinical[clinical['Exclusion'] == 0.0]
clinical = clinical[clinical['RFS_event'] != 'Unknown']

clinical['T0_medseq_TFE'] = clinical['T0_lcode'].map(lcode2tfe)
clinical['T3_medseq_TFE'] = clinical['T3_lcode'].map(lcode2tfe)


from vafpredictor import fitLinearPredictorOdds, predictLogOdds
datafit = clinical[clinical['T0_medseq_success'] == 1.0]
datafit = datafit[datafit['T0_VAF_oncomine'] > 0]

params = fitLinearPredictorOdds(datafit['T0_medseq_TFE'], datafit['T0_VAF_oncomine'])

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(datafit['T0_medseq_TFE'], datafit['T0_VAF_oncomine'])

m = np.min(datafit['T0_medseq_TFE'])
M = np.max(datafit['T0_medseq_TFE'])

x = np.linspace(m, M, 500)
y = predictLogOdds(x, params)
ax.plot(x, y, color='k')


ax.set_xlabel('medseq TFE', fontsize=14)
ax.set_ylabel('Oncomine VAF', fontsize=14)


clinical['T0_VAF_oncomine_predicted_from_medseq'] = predictLogOdds(clinical['T0_medseq_TFE'], params)


clinical.to_csv(args.outfile)
