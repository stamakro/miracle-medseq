# reproduce miracle medseq paper

RAWDBFILE='../../../data/clinical_database_final2_hgp.txt';
CLEANDBFILE='../data/cleanDB.csv';
MUTATIONSPCRFILE='/home/stavros/Desktop/code/nb-deconv/medseq-deconv/data/new_databestand_Stavros_240124.csv';
MEDSEQSAMPLEOVERZICHTCFDNA='../../../data/me/sampleoverzicht_SW_20240219.csv*';
MUTATIONSONCOMINEFILE='../../../data/dna/20241127_Oncomine_BL.txt';
HBDMEDSEQCOUNTS='../data/hbd_cpgi.csv';
CRLMMEDSEQCOUNTS='../data/hbd_cpgi.csv';
MIRACLEMEDSEQCOUNTS='/home/stavros/emc/projects/MedSeq/processed/miracle-latest-2024-06-23-11-25/counts_aggregated_cpgi.csv';
REGELDICT='../data/regulatory/cpgi_enhancers_promoters.pkl';
DMRCALLINGOUTPUTFILE='../results/dmrs_halfvariable_fwer_fc0_crlm_hbd_vafcor_in_exclusions.csv';
LCODETFEPKL='../results/lcode2tfe.pkl';
DATASET='../data/dataset.csv';

export PYTHONPATH=$PYTHONPATH:/home/stavros/Desktop/code/beta-kde/;


# where to store figures and intermediate results
mkdir  -p '../figures/';
mkdir  -p '../results/';

# step 0, read clinical DB
if ! [ -f '../data/cleanDB.csv' ];
then
    conda activate base;
    echo 'running initial data engineering from DB';
    python cleanClinicalDB.py --rawdbfile $RAWDBFILE --mutfile $MUTATIONSPCRFILE --miraclecfdnainfo $MEDSEQSAMPLEOVERZICHTCFDNA --mutfileoncomine $MUTATIONSONCOMINEFILE --outputfile $CLEANDBFILE;
fi;


# step 1, DMRs
# for rpy2
conda activate rpy;

# find DMRs tissue vs healthy cf, validate on exclusions
python dmrCalling.py --cleandb $CLEANDBFILE --hbdfile $HBDMEDSEQCOUNTS --tumorfile $CRLMMEDSEQCOUNTS --cfdnafile $MIRACLEMEDSEQCOUNTS --miraclecfdnainfo $MEDSEQSAMPLEOVERZICHTCFDNA --promoterdict $REGELDICT --output $DMRCALLINGOUTPUTFILE;

# deconvolve, get tfes and compare to VAF
# also make a dataframe with patients that can be used in further analysis
# removes AMP29 who has missing RFS
python deconvolve_standard.py --cleandb $CLEANDBFILE --hbdfile $HBDMEDSEQCOUNTS --tumorfile $CRLMMEDSEQCOUNTS --cfdnafile $MIRACLEMEDSEQCOUNTS --miraclecfdnainfo $MEDSEQSAMPLEOVERZICHTCFDNA --dmrfile $DMRCALLINGOUTPUTFILE --tfedict $LCODETFEPKL --output $DATASET;

conda deactivate;

# for survival analysis/lifelines/sklearn
conda activate mcmc;

python predict_t0.py --dataset $DATASET;

conda deactivate;

conda activate rpy;
python growthpatterns.py --dataset $DATASET --cfdnafile $MIRACLEMEDSEQCOUNTS --miraclecfdnainfo $MEDSEQSAMPLEOVERZICHTCFDNA;

conda deactivate;
