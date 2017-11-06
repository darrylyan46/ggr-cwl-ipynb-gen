# ipynb Generator configuration
DATA_SOURCES_SFTP   = 'sftp'
DATA_SOURCES_MISEQ  = 'miseq'
DATA_SOURCES_DUKEDS = 'dukeds'
DATA_SOURCES_OTHER  = 'other'
data_sources = [
    DATA_SOURCES_SFTP,
    DATA_SOURCES_MISEQ,
    DATA_SOURCES_OTHER,
    DATA_SOURCES_DUKEDS
]
library_type_chip_seq = 'chip_seq'
library_type_rna_seq = 'rna_seq'
library_type_atac_seq = 'atac_seq'
notebook_blurb = "This notebook will create all the necessary files, scripts and folders to pre-process " \
                 "the aforementioned project. Is designed to be used in a jupyter server deployed in a system running " \
                 "SLURM. The majority of the scripts and heavy-lifting processes are wrapped up in sbatch scripts." \
                 "As an end user, in order to pre-process your samples provided in the spread sheet, " \
                 "you will simply need to *run the entire notebook* (Cell > Run all) and the system should take care " \
                 "of the rest for you."

# Pipelines configuration
star_genome='/data/reddylab/Reference_Data/Genomes/hg38/STAR_genome_sjdbOverhang_49_novelSJDB'
separate_jsons=True
mem = {'chip_seq': 24000, 'rna_seq': 48000, 'atac_seq': 24000}
nthreads = {'chip_seq': 16, 'rna_seq': 24, 'atac_seq': 16}
seq_ends = ['se', 'pe']
peak_types = ['narrow', 'broad']
with_controls = [False, 'with-control']
strandnesses = ['unstranded', 'stranded', 'revstranded']
blacklist_removal = [None, 'blacklist-removal']
with_sjdb = True

# Environment configuration
conda_activate = '/data/reddylab/software/miniconda2/bin/activate'
contamination_script='/data/reddylab/Darryl/GitHub/reddylab/contamination_script'
plot_script = '/data/reddylab/Darryl/GitHub/reddylab/countFactors_metadata.sh'
qc_script_dir = '/data/reddylab/software/cwl/bin'
HOST_FOR_TUNNELED_DOWNLOAD = "Hardac-xfer.genome.duke.edu"
