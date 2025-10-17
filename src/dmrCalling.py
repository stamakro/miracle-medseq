import pandas as pd
import numpy as np
from rutils import *
import argparse
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
import pickle


def getChromosome(reg: str) -> str:
    return reg.split(':')[0]

parser = argparse.ArgumentParser(prog='dmrCalling.py', description='')

parser.add_argument('--clindb', dest='clindb', metavar='CLINICALDB', help='path to clean DB file', default='../data/cleanDB.csv')
parser.add_argument('--hbdfile', dest='hbdfile', metavar='HBDFILE', help='path to HBD methylation counts and sample info', default='../data/hbd_cpgi.csv')
parser.add_argument('--tumorfile', dest='tumorfile', metavar='TUMORFILE', help='path to tumor methylation counts and sample info', default='../data/crlm_cpgi.csv')
parser.add_argument('--cfdnafile', dest='miraclefile', metavar='CFDNAFILE', help='path to methylation counts of miracle samples', default='/home/stavros/emc/projects/MedSeq/processed/miracle-latest-2024-06-23-11-25/counts_aggregated_cpgi.csv')
parser.add_argument('--miraclecfdnainfo', dest='medseq_sample_info', metavar='INFOMEDSEQ', help='path to csv file containing CpG reads, mapping etc for MIRACLE cfdna samples', default='/home/stavros/Desktop/code/nb-deconv/medseq-deconv/data/new_sampleoverzicht_SW_20240219.csv')
parser.add_argument('--promoterdict', dest='promoter_dict', metavar='PROMOTERDICT', help='path to pkl file containing mapping to known REGELs', default='../data/regulatory/cpgi_enhancers_promoters.pkl')
parser.add_argument('--output', dest='outputfile', metavar='OUTPUTFILE', help='path to csv file containing DMRs, fold changes, etc', default='../results/dmrs_halfvariable_fwer_fc0_crlm_hbd_vafcor_in_exclusions.csv')


args = parser.parse_args()
sys.exit(0)

#env rpy
Ngenes = 500
logFC = 2.5


dataHbd = pd.read_csv(args.hbdfile, index_col=0)
dataTumor = pd.read_csv(args.tumorfile, index_col=0)

countsHbd = dataHbd.iloc[np.where(pd.Series(dataHbd.index).apply(isAutosomal))[0]].astype(float)
countsTumor = dataTumor.iloc[np.where(pd.Series(dataTumor.index).apply(isAutosomal))[0]].astype(float)


librarysizeHbd = np.round(np.array(dataHbd.loc['Used reads'].astype(float)))
librarysizeTumor = np.round(np.array(dataTumor.loc['Used reads'].astype(float)))


countsBloodTumor = pd.concat((countsHbd, countsTumor), axis=1)
librarysizeBloodTumor = np.hstack((librarysizeHbd, librarysizeTumor))


labels = np.ones(countsBloodTumor.shape[1], int)
labels[:countsHbd.shape[1]] = 0


mu = np.mean(countsBloodTumor, axis=1)
sigma = np.std(countsBloodTumor, axis=1, ddof=1)


ind = np.where(sigma > 0)[0]

mu = mu[ind]
sigma = sigma[ind]
countsBloodTumor = countsBloodTumor.iloc[ind]


cv = sigma / mu
threshold = np.median(cv)

ind = np.where(cv > threshold)[0]
countsBloodTumor = countsBloodTumor.iloc[ind]

dmrInfo = r_dmr_edgeR_tissue(countsBloodTumor.T, librarysizeBloodTumor, labels, maxMarkers=countsBloodTumor.shape[0], minFC=0.)
# in total 3,628 DMRs with Bonferroni correction

sys.exit(0)
###############################################################################
liverData = pd.read_csv('/home/stavros/emc/projects/MedSeq/processed/liver-2025-06-30-21-18/counts_aggregated_cpgi.csv',index_col=0)
countsLiver = liverData.loc[countsHbd.index]

dd = {'H17-2409': 14130193.0,
 'H94-3521': 7035922.0,
 'H18-20019': 13510083.0,
 'H91-5063': 13987352.0,
 'H91-7053': 13384933.0,
 'H91-14328': 13886509.0,
 'H17-381': 13669314.0,
 'H93-10410': 15617285.0,
 'H92-108': 12748987.0,
 'H94-3854': 12417421.0,
 'H12-16946': 13501276.0,
 'H14-7336': 14570170.0,
 'H14-5502': 16344821.0,
 'H92-10299': 14113936.0,
 'H18-1952': 13266928.0,
 'H91-12731': 14698250.0,
 'H93-4355': 13164478.0,
 'H94-7144': 14061417.0,
 'H94-5429': 12579277.0,
 'H92-14906': 14154025.0,
 'H92-2866': 14112683.0,
 'H94-7789': 8745985.0,
 'H18-10766': 14410008.0,
 'H18-10573': 13319707.0,
 'H94-6176': 14160397.0,
 'H08-22541': 15444586.0,
 'H94-2181': 14232665.0,
 'H92-2816': 13543655.0,
 'H17-1277': 13474133.0,
 'H92-2814': 7720703.0,
 'H12-24094': 13156419.0,
 'H93-8167': 9750525.0}

librarysizeLiver = np.array(countsLiver.columns.map(dd))

countsBloodLiver = pd.concat((countsHbd, countsLiver), axis=1)
librarysizeBloodLiver = np.hstack((librarysizeHbd, librarysizeLiver))


labels2 = np.ones(countsBloodLiver.shape[1], int)
labels2[:countsHbd.shape[1]] = 0


mu2 = np.mean(countsBloodLiver, axis=1)
sigma2 = np.std(countsBloodLiver, axis=1, ddof=1)


ind = np.where(sigma2 > 0)[0]

mu2 = mu2[ind]
sigma2 = sigma2[ind]
countsBloodLiver = countsBloodLiver.iloc[ind]


cv2 = sigma2 / mu2
threshold = np.median(cv2)

ind = np.where(cv2 > threshold)[0]
countsBloodLiver = countsBloodLiver.iloc[ind]

dmrInfoLiver = r_dmr_edgeR_tissue(countsBloodLiver.T, librarysizeBloodLiver, labels2, maxMarkers=countsBloodLiver.shape[0], minFC=0.)

dmrInfo = dmrInfo.loc[np.setdiff1d(dmrInfo.index, dmrInfoLiver.index)]

###############################################################################
# get the 11 excluded patients that had an oncomine hit
clinical = pd.read_csv(args.clindb, index_col=0)
clinical = clinical[clinical['Exclusion'] == 1.0]
clinical = clinical[clinical['T0_medseq_success'] == 1.0]
clinical = clinical[clinical['T0_VAF_oncomine'] > 0.0]
clinical = clinical[clinical['Reason exclusion'] != 'PA: levermetastasen van slokdarmCa']

miracleData = pd.read_csv(args.miraclefile, index_col=0)

excludedData = miracleData[clinical['T0_lcode']]

miracleinfo = pd.read_csv(args.medseq_sample_info, index_col=0)
miracleinfo = miracleinfo.loc[clinical['T0_lcode']]

excludedTPM = counts2tpm(excludedData)
excludedLogTPM = np.log(excludedTPM + 1.0)


dmrdata = np.array(excludedLogTPM.loc[dmrInfo.index].T)
vaf = np.array(clinical['T0_VAF_oncomine'])

rhos = np.zeros(dmrInfo.shape[0])
ps = np.zeros(rhos.shape)
psOnesided = np.zeros(rhos.shape)

for i in range(dmrInfo.shape[0]):
    rhos[i], ps[i] = spearmanr(vaf, dmrdata[:, i])
    if dmrInfo.iloc[i]['logFC'] > 0:
        _, psOnesided[i] = spearmanr(vaf, dmrdata[:,i], alternative='greater')
    else:
        _, psOnesided[i] = spearmanr(vaf, dmrdata[:,i], alternative='less')

top500 = dmrInfo[dmrInfo['logFC'].abs()>logFC].iloc[:Ngenes]
isInSelection = np.zeros(dmrInfo.shape[0])

for i, reg in enumerate(dmrInfo.index):
    if reg in top500.index:
        isInSelection[i] = 1.


rhos[np.isnan(rhos)] = 0.0
ps[np.isnan(ps)] = 1.0
psOnesided[np.isnan(psOnesided)] = 1.0


pcor = multipletests(ps, method='fdr_bh')[1]
pcorOnesided = multipletests(psOnesided, method='fdr_bh')[1]



fig = plt.figure()
ax = fig.add_subplot(1,3,1)

ax.hist2d(dmrInfo['logFC'], rhos, bins=[50,50], density=True, cmap='Blues')

ax = fig.add_subplot(1,3,2)
ax.scatter(dmrInfo['logFC'].iloc[np.where(pcor < 0.05)[0]], rhos[pcor < 0.05], color='C1', label='significant')
ax.scatter(dmrInfo['logFC'].iloc[np.where(pcor >= 0.05)[0]], rhos[pcor >= 0.05], color='k', alpha=0.5)

ax.axvline(2.5, color='C2', linestyle='--')
ax.axvline(-2.5, color='C2', linestyle='--')

ax = fig.add_subplot(1,3,3)
ax.scatter(dmrInfo['logFC'].iloc[np.where(isInSelection)[0]], rhos[np.where(isInSelection)[0]], color='C1', label='selected for deconvo')
ax.scatter(dmrInfo['logFC'].iloc[np.where(1-isInSelection)[0]], rhos[np.where(1-isInSelection)[0]], color='k', alpha=0.5)

ax.axvline(2.5, color='C2', linestyle='--')
ax.axvline(-2.5, color='C2', linestyle='--')

fig.savefig('../figures/DMRS_foldchange_vafcor.png', dpi=600)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.scatter(dmrInfo['logFC'].iloc[np.where(pcor < 0.05)[0]], rhos[pcor < 0.05], color='C1',  s=10, alpha=0.5, label='significant')
ax.scatter(dmrInfo['logFC'].iloc[np.where(pcor >= 0.05)[0]], rhos[pcor >= 0.05], color='k', s=10, alpha=0.5)

ax.set_xlabel('log fold-change, tumor tissue vs healthy cfDNA')
ax.set_ylabel('Spearman correlation methylation profile and VAF')

ax.legend()

fig.savefig('../figures/final_DMRS_foldchange_vafcor.png', dpi=1200)
fig.savefig('../figures/final_DMRS_foldchange_vafcor.svg', dpi=600)







dmrInfo['correlation_with_VAF_exclusion'] = rhos
dmrInfo['FDR_correlation_with_VAF_exclusion'] = pcor
dmrInfo['FDR_correlation_with_VAF_exclusion_sided'] = pcorOnesided

dmrInfo['logFC_correlation_sign_matches'] = (dmrInfo['correlation_with_VAF_exclusion'] * dmrInfo['logFC'] > 0).astype(int)

dmrInfo['chromosome'] = pd.Series(np.array(pd.Series(dmrInfo.index).apply(getChromosome)), index=dmrInfo.index)


chromosomes, sitesperchromosome = np.unique(pd.Series(dataTumor.index[:27923]).apply(getChromosome), return_counts=True)

chrom2Nregions = dict()
for c, n in zip(chromosomes, sitesperchromosome):
    chrom2Nregions[c] = n

chrom2NregionsDMR = dmrInfo.groupby('chromosome').count()['logFC'].to_dict()

# for chromosome distribution of those with VAF correlation
chrom2NregionsDMRandVAF = dmrInfo.iloc[np.where(np.logical_and(dmrInfo['FDR_correlation_with_VAF_exclusion'] < 0.05, dmrInfo['logFC_correlation_sign_matches']))[0]].groupby('chromosome').count()['logFC'].to_dict()
NislandsFurtherSelection = np.sum(np.logical_and(dmrInfo['FDR_correlation_with_VAF_exclusion'] < 0.05, dmrInfo['logFC_correlation_sign_matches']))

# # if one-sided
# chrom2NregionsDMRandVAF = dmrInfo.iloc[np.where(dmrInfo['FDR_correlation_with_VAF_exclusion_sided'] < 0.05)[0]].groupby('chromosome').count()['logFC'].to_dict()
# NislandsFurtherSelection = np.sum(dmrInfo['FDR_correlation_with_VAF_exclusion_sided'] < 0.05)


chroms = np.arange(1,23)
sitesPerChrom = np.zeros(chroms.shape[0], int)
Nislands = 0
for k,v in chrom2Nregions.items():
    if k != 'chrX' and k != 'chrY':
        Nislands += v

expectedPerChrom = np.zeros(chroms.shape[0])
sitesPerChromEnrich = np.zeros(chroms.shape[0])
sitesPerChrom = np.zeros(chroms.shape[0])

sitesPerChrom2 = np.zeros(chroms.shape[0])
sitesPerChromEnrich2 = np.zeros(chroms.shape[0])
expectedPerChrom2 = np.zeros(chroms.shape[0])

for i in chroms:
    sitesPerChrom[i-1] = chrom2NregionsDMR['chr'+str(i)]
    expectedPerChrom[i-1] = dmrInfo.shape[0] * chrom2Nregions['chr'+str(i)] / Nislands
    sitesPerChromEnrich[i-1] = sitesPerChrom[i-1] / expectedPerChrom[i-1]

    sitesPerChrom2[i-1] = chrom2NregionsDMRandVAF['chr'+str(i)]
    expectedPerChrom2[i-1] = NislandsFurtherSelection * chrom2Nregions['chr'+str(i)] / Nislands
    sitesPerChromEnrich2[i-1] = sitesPerChrom2[i-1] / expectedPerChrom2[i-1]



fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(1,2,1)

ax.bar(chroms, sitesPerChromEnrich)
# ax.bar(chroms, sitesPerChromRel*100)
# ax.axhline(np.median(sitesPerChromRel)*100., color='k', linestyle='--', label='median')

ax.set_xticks(chroms)

ax.axhline(1.0, color='k', linestyle='--')

ax.set_xlabel('chromosomes', fontsize=16)
ax.set_ylabel('Enrichment of DMRs', fontsize=16)
ax.set_title('%d DMRs tissue vs healthy cfdna' % dmrInfo.shape[0])


ax = fig.add_subplot(1,2,2)

ax.bar(chroms, sitesPerChromEnrich2)

ax.set_xticks(chroms)

ax.axhline(1.0, color='k', linestyle='--')

ax.set_xlabel('chromosomes', fontsize=16)
ax.set_ylabel('Enrichment of DMRs', fontsize=16)
ax.set_title('%d DMRs validated in exclusions' % NislandsFurtherSelection)
plt.tight_layout()

fig.savefig('../figures/DMRS_chromosome_enrichment', dpi=600)


with open(args.promoter_dict, 'rb') as f:
    regulatoryRegions = pickle.load(f)


dmrInfo['promoter'] = dmrInfo.index.map(regulatoryRegions['promoter'])
dmrInfo['cisenhancer_highconf'] = dmrInfo.index.map(regulatoryRegions['enhancer_high_conf'])
dmrInfo['cisenhancer_lowconf'] = dmrInfo.index.map(regulatoryRegions['enhancer_low_conf'])


dmrInfo.to_csv(args.outputfile)
