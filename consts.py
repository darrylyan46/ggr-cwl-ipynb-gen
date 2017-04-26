library_type_chip_seq = 'chip_seq'
library_type_rna_seq = 'rna_seq'
chip_seq_pipeline_types=[
    'se-narrow',
    'se-narrow-with-control',
    'pe-narrow',
    'pe-narrow-with-control',
    'se-broad',
    'se-broad-with-control',
    'pe-broad',
    'pe-broad-with-control'
]
rna_seq_pipeline_types=[
    'pe-unstranded',
    'pe-stranded',
    'pe-revstranded',
    'pe-unstranded-with-sjdb',
    'pe-stranded-with-sjdb',
    'pe-revstranded-with-sjdb',
    'se-unstranded',
    'se-stranded',
    'se-revstranded',
    'se-unstranded-with-sjdb',
    'se-stranded-with-sjdb',
    'se-revstranded-with-sjdb'
]
star_genome='/data/reddylab/Reference_Data/Genomes/hg38/STAR_genome_sjdbOverhang_49_novelSJDB'
separate_jsons=True
mem = {'chip_seq': 32000, 'rna_seq': 48000}
nthreads = {'chip_seq': 16, 'rna_seq': 24}
seq_ends = ['se', 'pe']
peak_types = ['narrow', 'broad']
with_controls = [False, 'with-control']
strandnesses = ['unstranded', 'stranded', 'revstranded']
with_sjdb = True