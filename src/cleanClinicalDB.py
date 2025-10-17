import pandas as pd
import numpy as np
from datetime import datetime
import argparse

def getPatIDoncomine(s: str):
    if s[-2:] != 'T0':
        return np.nan

    return s[:-2]


def processid(x: str):
    # id is a str hospital code and a number
    numberstart = len(x) - 1

    # start at the end and keep looking for the start of the number
    while numberstart > 0:
        try:
            a = int(x[numberstart-1:])
            numberstart -= 1

        except ValueError:
            # found a letter
            break


    hospital = x[:numberstart]
    nr = x[numberstart:]

    # M was switched to EMC and 0-padded to be 3 digits
    if hospital == 'M':
        hospital = 'EMC'
        if int(nr) < 10:
            nr = '00' + nr
        elif int(nr) < 100:
            nr = '0' + nr

    # some IJsselland pats had padding to be 3 digits and that was removed
    if hospital == 'YSL':
        if int(nr) > 9 and nr[0] == '0':
            nr = nr[1:]

        # else:
        #     nr = '0' + nr


    return (hospital + nr)



def calculateDifference(entry, cs, ce):

    if type(entry[cs]) is float:
        assert np.isnan(entry[cs])
        return np.nan

    if type(entry[ce]) is float:
        assert np.isnan(entry[ce])
        return np.nan

    start = datetime.strptime(entry[cs], '%d-%m-%Y')
    end = datetime.strptime(entry[ce], '%d-%m-%Y')
    diff = end - start

    return diff.days

def convertNumerical(data, thres: float, aboveIsTrue: bool):
    result = np.zeros(data.shape[0], float)
    for i,s in enumerate(data):
        if np.isnan(s):
            result[i] = np.nan

        else:
            decision = int(s >= thres)
            if aboveIsTrue:
                result[i] = decision
            else:
                result[i] = 1 - decision

    return result


def convertStringBinary(data: pd.Series, target: str):
    result = np.zeros(data.shape[0], float)
    for i,s in enumerate(data):
        try:
            if np.isnan(s):
                result[i] = np.nan
        except TypeError:
            assert type(s) is str
            result[i] = int(s == target)
    return result







if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='cleanClinicalDB.py', description='')

    parser.add_argument('--rawdbfile', dest='db', metavar='RAWDB', help='path to raw DB file copied from miracle folder', default='../../../data/clinical_database_final3_20250623.txt')
    parser.add_argument('--rawdbfile2', dest='dbold', metavar='RAWDB', help='every time there\'s an update some columns are changed...', default='../../../data/clinical_database_final2_hgp.txt')
    parser.add_argument('--mutfile', dest='mutation_pcr_file', metavar='PCRFILE', help='path to csv file containing PCR vafs and CTC counts per timepoint', default='/home/stavros/Desktop/code/nb-deconv/medseq-deconv/data/new_databestand_Stavros_240124.csv')
    parser.add_argument('--miraclecfdnainfo', dest='medseq_sample_info', metavar='INFOMEDSEQ', help='path to csv file containing CpG reads, mapping etc for MIRACLE cfdna samples', default='/home/stavros/Desktop/code/nb-deconv/medseq-deconv/data/new_sampleoverzicht_SW_20240219.csv')
    parser.add_argument('--mutfileoncomine', dest='mutation_oncomine_file', metavar='ONCOMINEFILE', help='path to csv file containing Oncomine VAFs for MIRACLE cfdna samples', default='../../../data/dna/20241127_Oncomine_BL.txt')
    parser.add_argument('--outputfile', dest='savefile', metavar='OUTPUTFILE', help='path to write clean csv file', default='../data/cleanDB.csv')


    args = parser.parse_args()

    dataOld = pd.read_table(args.dbold, index_col=0)

    if dataOld.iloc[-1].isna().all():
        dataOld = dataOld.iloc[:-1]


    dataOld = dataOld[['pushing', 'replacement', 'desmoplastic', 'HGP']]


    data = pd.read_table(args.db, index_col=0)


    if data.iloc[-1].isna().all():
        data = data.iloc[:-1]

    # remove columns with clinical notes for a handful of patients
    assert (data.iloc[:,-7:].isna().mean() > 0.91).all()

    data = data.iloc[:,:-7]

    assert (data.index == dataOld.index).all()
    data = pd.concat((data, dataOld), axis=1)

    data.drop(['Patient Identification Number EMC', 'Date_of_birth', 'Date_resection_primary'], axis=1, inplace=True)


    data['OS_days'] = data.apply(calculateDifference, axis=1, args=('Date_resection_CRLM','date death or followup date'))


    renamedict = {
    'Zkh': 'hospital',
    'Age_resection_CRLM': 'age',
    'Recurrence1': 'RFS_event',
    'Recurrence_DFI_days': 'RFS_days',
    'CEA _before_resection_CRLM': 'CEA_before_resection_CRLM' ,
    'CEA _before_resection_CRLM_cat': 'CEA_before_resection_CRLM_cat',
    'CEA _before_resection_CRLM_cat2': 'CEA_before_resection_CRLM_cat2',
    'Number_CRLM ': 'Number_CRLM'
    }

    data.rename(renamedict, axis=1, inplace=True)


    data['max1y_primary_CRLM'] = convertStringBinary(data['DFI _cat'], '=/<1 year')
    data['isN0'] = convertNumerical(data['N_stage_CRC'], 1.0, False)
    # nr_crlm_cat is radiology, cat2 is pathology
    # radiology is used for fong score
    data['Two_or_more_CRLMs'] = convertStringBinary(data['Number_CRLM_cat'], '>1')
    data['CEA_high'] = convertStringBinary(data['CEA_before_resection_CRLM_cat2'], '>5')
    data['CEA_veryhigh'] = convertStringBinary(data['CEA_before_resection_CRLM_cat'], '>200')
    data['diameter5plus'] = convertNumerical(data['Diameter_largest_CRLM'], 5.0, True)
    data['isR1'] = convertStringBinary(data['Resection_margin-CRLM'], 'R1')
    data['OS_event'] = convertStringBinary(data['death yes no'], 'Yes')

    # this is the radiology fong score, the original one
    data['fongHigh'] = convertStringBinary(data['Riskgroup_fong'], 'High') + convertStringBinary(data['Riskgroup_fong'], 'High ')
    data['isMale'] = convertStringBinary(data['Gender'], 'Male')
    data['Exclusion'] = convertStringBinary(data['Exclusion'], 'Yes')
    data['metachronous'] = convertStringBinary(data['Meta_syn'], 'Metachronous')
    data['rightsided_primary'] = convertStringBinary(data['Location_primary'], 'Right-sided')
    data['rectal_primary'] = convertStringBinary(data['Location_primary'], 'Rectum')

    data['multiorgan_recurrence'] = convertStringBinary(data['Multiorgan'], 'Yes')
    data['Additional_therapy_metastases'] = convertStringBinary(data['Additional_therapy_metastases'], 'Yes')

    data.drop(['DFI _cat', 'N_stage_CRC', 'Number_CRLM_cat', 'Number_CRLM_cat2', 'CEA_before_resection_CRLM_cat', 'CEA_before_resection_CRLM_cat2', 'Diameter_largest_CRLM_cat', 'Diameter_largest_CRLM_cat2', 'Riskgroup_fong', 'Riskgroup_fong_pathology', 'Date_resection_CRLM', 'Comorbidity',
    'Comorbidity_cat', 'Comorbidity_specific', 'ASA_class', 'Resection_type', 'Preoperative_radiotherapy_primary', 'Type_rtx_primary','Operation_type_primary', 'T_stage_CRC', 'Lymphnodes', 'Differentiation_CRC', 'Radicality_CRC', 'Tissue used for genesequencing', 'Tissue used for MSI determination', 'KRAS_mstatus','NRAS_status','HRAS_status', 'BRAF_status','PIK3CA_status','P53_status','APC_status','MSI_status','Notes','PA-number of tissue used for geneseqencing',
    'PA-number of tissue used for MSI determination','Date of genesequencing','Date of MSI determination', 'Mutation_KRAS','Mutation_NRAS','Mutation_HRAS','Mutation_BRAF','Mutation_PIK3CA','Mutation_P53','Mutation_APC', 'Notes.1','Other','Adjuvant_ctx_CRC','Type_adjuvant_ctx','Date_detection_CRLM', 'Gender', 'Fong_score_radiology', 'With any CEA preop (not only prior CTx)','Number of unknown FONG items','Distribution_CRLM','CRLM_location',
    'Number_locations', 'Number_segments_wig','Which_segments_wig','Number_segments_total','Which_segments_total','3_more_segments_resected','Hemihepatectomy','Number_CRLM_in_pathology','Largest_diameter_CRLM_pathology','Distance_resection_margin','Resection_margin-CRLM', 'RFA/MWA/SRx/ILP', 'Number CRLM treated with RFA/microwave/SRx', 'Specify type of additional therapy. In RFA/MWA (lokation, number)','Total_laesions_treated_CRLM', 'Days_hospitalisation_CRLM','Complication_resection _CRLM','Death_postop','Dindo_cat','Extrahepatic yes/no','EHD preoperative yes/no','Extrahepatic where','EHD_text','Date_recurrence1', 'Exact_location_recurrence','Location_recurrence','Rec1_Solitary','Rec1_Solitary_specify','Multiorgan','Intra_and_extrahepatic_recurrence','Intrahepatic_only','Extrahepatic_only','Intrahepatic_with_or_without_extrahepatic','Extrahepatic_with_or_without_extrahepatic',
    'Pulmonary_with_or_without_else','Local_recurrence_with_or_without_else','Distant_lymphnodes_with_or_without_else','Peritonitis_with_or_without_else','Bone_with_or_without_else','Ovary_with_or_without_else','Adrenal_with_or_without_else','Brain_with_or_without_else','Including_ovary_adrenal_brain','Recurrence1_palliative_curative', 'death yes no', 'date death or followup date','Date of updated disease status or entry into database','MiracleNr', 'Tumor In Situ EHD yes/no',
    'Tumor In Situ CRLM yes/no', 'Meta_syn', 'Location_primary'], axis=1, inplace=True)

    multivariate = ['age', '>1y from primary to CRLM', 'isN0', '2plus_CRLMs', 'CEA_high', 'diameter5plus', 'isR1', 'OS_days', 'OS_event']

    #
    dfmutctc = pd.read_csv(args.mutation_pcr_file, index_col=0, delimiter=';')


    dfmutctc.iloc[np.where(dfmutctc['VAF @T3'] == 'TBD')[0], -2] = np.nan


    for p in dfmutctc.index:
        assert p in data.index

    infomutctc = dfmutctc.to_dict()


    data['T0_VAF'] = pd.Series(data.index, index=data.index).map(infomutctc['VAF @T0']) / 100.
    data['T3_VAF'] = pd.Series(data.index, index=data.index).map(infomutctc['VAF @T3']).astype(float) / 100.
    data['T0_CTCs'] = pd.Series(data.index, index=data.index).map(infomutctc['nr of CTCs @BL'])
    data['T3_CTCs'] = pd.Series(data.index, index=data.index).map(infomutctc['nr of CTCs @T3'])

    # 4 vafs still missing/tbd at T3
    #      AMP04, AMP27, EMC107: CHIP
    # YSL45: T3 0, T0: 4.409%
    # ASZ23: T3 0, T0: 1%

    t0i = np.where(data.columns == 'T0_VAF')[0][0]
    t3i = np.where(data.columns == 'T3_VAF')[0][0]

    for p in ['AMP04', 'AMP27', 'EMC107']:

        y = np.where(data.index == p)[0][0]
        print(y)
        data.iloc[y,t0i] = np.nan
        data.iloc[y,t3i] = np.nan

    for p in ['YSL45', 'ASZ23']:
        y = np.where(data.index == p)[0][0]
        data.iloc[y,t3i] = 0.



    medseqinfo = pd.read_csv(args.medseq_sample_info, index_col=0)

    medseqinfo = medseqinfo[medseqinfo['Timepoint'] != 1]
    medseqinfo = medseqinfo[medseqinfo['Timepoint'] != 2]
    medseqinfo['MIRACLE_ID'] = medseqinfo['Pat Name'].apply(processid)

    medseqinfo['QC_PASS'] = (medseqinfo['Used reads'] > 3e6).astype(int) * (medseqinfo['% filtered reads'] > 20.).astype(int)



    pat2l0 = dict()
    pat2l3 = dict()

    for p in data.index:
        assert p not in pat2l0
        assert p not in pat2l3

        tmp = medseqinfo[medseqinfo['MIRACLE_ID'] == p]

        if tmp.shape[0] == 0:
            pat2l0[p] = np.nan
            pat2l3[p] = np.nan

        else:
            ll = tmp.index[tmp['Timepoint'] == 0]
            assert len(ll) < 2

            if len(ll) == 0:
                pat2l0[p] = np.nan
            else:
                pat2l0[p] = ll[0]

            ll = tmp.index[tmp['Timepoint'] == 3]
            assert len(ll) < 2

            if len(ll) == 0:
                pat2l3[p] = np.nan
            else:
                pat2l3[p] = ll[0]


    data['T0_lcode'] = data.index.map(pat2l0)
    data['T3_lcode'] = data.index.map(pat2l3)

    data['T0_medseq_success'] = data['T0_lcode'].map(medseqinfo['QC_PASS'].to_dict())
    data['T3_medseq_success'] = data['T3_lcode'].map(medseqinfo['QC_PASS'].to_dict())

    oncomine = pd.read_csv(args.mutation_oncomine_file, delimiter=';',header=None)
    # pt code+T, lcode, duplicate, #muts, maxvaf, blabla
    oncomine.rename({0: 'analysisID', 1: 'LcodeDNA', 2: 'isDuplicate', 3: 'nrMutations', 4: 'T0_VAF_oncomine_raw'}, axis=1, inplace=True)

    oncomine['T0_VAF_oncomine'] = oncomine['T0_VAF_oncomine_raw'].apply(lambda x: 0.01 * float(x.replace(',','.')))


    oncomine['patID'] = oncomine['analysisID'].apply(getPatIDoncomine)

    oncomine = oncomine[~oncomine['patID'].isna()]
    pat2oncomine = oncomine.set_index('patID')['T0_VAF_oncomine'].to_dict()

    data['T0_VAF_oncomine'] = data.index.map(pat2oncomine)

    ######
    # additional cleaning up
    data['Oneyear_OS_event'] = convertStringBinary(data['Oneyear_OS_event'], 'Yes')
    data['RFS_event'] = convertStringBinary(data['RFS_event'], 'Yes')

    data['HGP_fully_desmoplastic'] = convertStringBinary(data['HGP'], 'dHGP')
    data['HGP_desmo_five'] = convertNumerical(data['desmoplastic'], 95.0, True)

    data.drop(['HGP'], axis=1, inplace=True)

    # play around a bit with data
    datam = data[data['Exclusion'] == 0]

    datam.drop(['Exclusion', 'Reason exclusion'], axis=1, inplace=True)


    datam2 = datam[~datam['T0_VAF'].isna()]

    missing = dict()
    for c in datam2.columns:
        if datam2[c].isna().any():
            missing[c] = set(list(datam2.index[np.where(datam2[c].isna())[0]]))

    datameth = datam[~datam['T0_lcode'].isna()]
    datameth = datameth[datameth['T0_medseq_success'] == 1.0]

    for c in datameth.columns:
        if datameth[c].isna().any():
            if c in missing:
                missing[c] = missing[c].union(set(list(datameth.index[np.where(datameth[c].isna())[0]])))
            else:
                missing[c] = set(list(datameth.index[np.where(datameth[c].isna())[0]]))

    # datameth.to_csv('../data/medseq_dataset_saskia.csv')

    dataex = data[data['Exclusion'] == 1]
    dataex = dataex[dataex['T0_medseq_success'] == 1.0]

    dataex = dataex[dataex['T0_VAF_oncomine']>0.]
    dataex.drop(['ASZ16'], axis=0, inplace=True)

    data.to_csv(args.savefile)
