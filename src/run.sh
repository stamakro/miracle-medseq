# reproduce miracle medseq paper

# some of these files contain sensitive data, including clinical data and survival outcomes
# they can be made available upon reasonable request
# database with clinical data
CLEANDBFILE='../data/cleanDB.csv';
# file with experiment and QC details
MEDSEQSAMPLEOVERZICHTCFDNA='../data/sampleoverzicht_SW_20240219.csv*';
# oncomine results
MUTATIONSONCOMINEFILE='../data/20241127_Oncomine_BL.txt';
# medseq read counts for healthy
HBDMEDSEQCOUNTS='../data/hbd_cpgi.csv';
# medseq read counts for CRLM tissues
CRLMMEDSEQCOUNTS='../data/crlm_cpgi.csv';
# medseq read counts for cfDNA of MIRACLE patients
MIRACLEMEDSEQCOUNTS='../data/counts_aggregated_cpgi.csv';
# match each CpG island to annotations for known promoters and enhancers
REGELDICT='../data/regulatory/cpgi_enhancers_promoters.pkl';
# where to save the DMRs
DMRCALLINGOUTPUTFILE='../results/dmrs_halfvariable_fwer_fc0_crlm_hbd_vafcor_in_exclusions.csv';
# where to save TFE-ME values
LCODETFEPKL='../results/lcode2tfe.pkl';
# where to save final dataset
DATASET='../data/dataset.csv';

export PYTHONPATH=$PYTHONPATH:/home/stavros/Desktop/code/beta-kde/;


# where to store figures and intermediate results
mkdir  -p '../figures/';
mkdir  -p '../results/';



# step 1, DMRs
# r-base 4.1.3, rpy2 3.5.12, scikit-learn 1.3.2, numpy 1.24.3
conda activate rpy;

# find DMRs tissue vs healthy cf, validate on exclusions
python dmrCalling.py --cleandb $CLEANDBFILE --hbdfile $HBDMEDSEQCOUNTS --tumorfile $CRLMMEDSEQCOUNTS --cfdnafile $MIRACLEMEDSEQCOUNTS --miraclecfdnainfo $MEDSEQSAMPLEOVERZICHTCFDNA --promoterdict $REGELDICT --output $DMRCALLINGOUTPUTFILE;

# deconvolve, get tfes and compare to VAF
# also make a dataframe with patients that can be used in further analysis
# removes patients who have missing RFS
python deconvolve_standard.py --cleandb $CLEANDBFILE --hbdfile $HBDMEDSEQCOUNTS --tumorfile $CRLMMEDSEQCOUNTS --cfdnafile $MIRACLEMEDSEQCOUNTS --miraclecfdnainfo $MEDSEQSAMPLEOVERZICHTCFDNA --dmrfile $DMRCALLINGOUTPUTFILE --tfedict $LCODETFEPKL --output $DATASET;

conda deactivate;

# for survival analysis/lifelines/sklearn
conda activate mcmc;

python predict_t0.py --dataset $DATASET;

conda deactivate;

conda activate rpy;
python growthpatterns.py --dataset $DATASET --cfdnafile $MIRACLEMEDSEQCOUNTS --miraclecfdnainfo $MEDSEQSAMPLEOVERZICHTCFDNA;

conda deactivate;
