import argparse
import nbformat
import nbformat.v3 as nbf
import sys
import os
import pandas as pd
from xlrd import XLRDError
import ruamel.yaml
import consts
import jinja2
import inspect
import numpy as np


def render(tpl_path, context):
    path, filename = os.path.split(tpl_path)
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(path or './')
    ).get_template(filename).render(context)


class Cell(object):
    def __init__(self, contents, description=None):
        self.contents = contents
        self.description = description
        if type(self.description) is not list:
            self.description = [self.description]
        self.header = []
        # self.header_inputs = []
        # self.header_outputs = []

    def writefile_to(self, dest):
        self.header = ["%%%%writefile %s" % dest]

    def to_list(self):
        cells = []
        if self.description:
            cells.append(nbf.new_text_cell('markdown', source=self.description))
        cells.append(nbf.new_code_cell(input=self.header + self.contents))
        return cells


class CellSbatch(Cell):
    def __init__(self, script_output='/dev/null', depends_on=False, mem=None, cpus=None,
                 partition=None, wrap=True, wrap_command='sh', array=None, **kwargs):
        super(CellSbatch, self).__init__(**kwargs)

        content_prolog = ['sbatch', '-o', script_output]
        if partition:
            content_prolog.extend(['-p', partition])
        if mem:
            content_prolog.extend(['--mem', str(mem)])
        if cpus:
            content_prolog.extend(['-c', str(cpus)])
        if depends_on:
            content_prolog.extend(['--depend', 'afterok:$1'])
        if array is not None:
            content_prolog.extend(['--array', array])
            wrap = False
        if wrap:
            content_prolog.append('--wrap="%s' % wrap_command)
            self.contents.append('"')
        self.contents = content_prolog + self.contents
        self.contents = [' '.join(self.contents)]

        self.header = ["%%script"]
        self.header.append('--out blocking_job_str')
        self.header.append("bash")

        if depends_on:
            self.header.append('-s "$blocking_job"')
        self.header = [' '.join(self.header)]

    def to_list(self):
        cells = super(CellSbatch, self).to_list()

        # We need to add an extra code cell to compute the SLURM job id
        extra_cell = Cell(
            contents=["import re", "blocking_job = re.match('Submitted batch job (\d+).*', blocking_job_str).group(1)"],
            description="Extract blocking job id"
        )
        cells.extend(extra_cell.to_list())
        return cells


def save_metadata(samples_df, conf_args, lib_type):
    cells = []
    cell_mkdir = Cell(contents=["%%bash",
                                "mkdir -p %s/data/%s/metadata" % (conf_args['root_dir'], lib_type),
                                "mkdir -p %s/data/%s/raw_reads" % (conf_args['root_dir'], lib_type),
                                "mkdir -p %s/data/%s/processed_raw_reads" % (conf_args['root_dir'], lib_type),
                                "mkdir -p %s/processing/%s/scripts" % (conf_args['root_dir'], lib_type),
                                "mkdir -p %s/processing/%s/jsons" % (conf_args['root_dir'], lib_type)
                                ],
                      description=["# %s - %s" % (conf_args['project_name'], lib_type),
                                   consts.notebook_blurb,
                                   "#### Create necessary folder(s)"])
    cells.extend(cell_mkdir.to_list())


    outfile = "%s/data/%s/metadata/%s_download_metadata.%s.txt" % (conf_args['root_dir'], lib_type, lib_type,
                                                                   conf_args['project_name'])
    contents = ["%%%%writefile %s" % outfile, samples_df.to_csv(index=False, sep=conf_args['sep'], encoding='utf-8')]
    cell = Cell(contents=contents, description="Save metadata file")
    cells.extend(cell.to_list())
    return cells, outfile


def download_fastq_files(conf_args, lib_type, metadata_filename=None):
    cells = []

    download_fn = "%s/processing/%s/scripts/download_%s.txt" % (conf_args['root_dir'], lib_type,
                                                                conf_args['project_name'])
    context = {
        'output_fn': download_fn,
        'project_name': conf_args['project_name'],
        'metadata_filename': metadata_filename,
        'root_dir': conf_args['root_dir'],
        'lib_type': lib_type
    }
    contents = [render('templates/download_fastq_files.j2', context)]

    cell_write_dw_file = Cell(contents=contents,
                              description=["#### Download FASTQ from sequencing core",
                                           "Create file to download FASTQ files from sequencing FTP"])
    cells.extend(cell_write_dw_file.to_list())

    execute_cell = CellSbatch(contents=['ssh hardac-xfer.genome.duke.edu;', 'sh %s' % download_fn],
                              wrap_command='',
                              description=" Execute file to download files")
    cells.extend(execute_cell.to_list())

    return cells


def ungzip_fastq_files(conf_args, lib_type, metadata_filename=None, num_samples=None):
    cells = []
    ungzip_fn = "%s/processing/%s/scripts/ungzip_%s.sh" % (conf_args['root_dir'], lib_type, conf_args['project_name'])
    context = {
        'output_fn' : ungzip_fn,
        'metadata_filename': metadata_filename,
        'project_name': conf_args['project_name'],
        'root_dir': conf_args['root_dir'],
        'lib_type': lib_type,
        'num_samples': num_samples
    }
    contents = [render('templates/ungzip_fastq_files.j2', context)]

    cell_write_dw_file = Cell(contents=contents, description="#### Ungzip FASTQ files")
    cells.extend(cell_write_dw_file.to_list())

    execute_cell = CellSbatch(contents=[ungzip_fn],
                              description="Execute file to ungzip FASTQ files",
                              depends_on=True,
                              array="0-%d%%20" % (num_samples - 1),
                              script_output="%s_%%A_%%a.out" % inspect.stack()[0][3])
    cells.extend(execute_cell.to_list())

    return cells


def merge_fastq_files(conf_args, lib_type, metadata_filename=None, num_samples=None):
    cells = []
    merge_fn = "%s/processing/%s/scripts/merge_lanes_%s.sh" % (conf_args['root_dir'], lib_type, conf_args['project_name'])
    context = {
        'output_fn' : merge_fn,
        'metadata_filename': metadata_filename,
        'project_name': conf_args['project_name'],
        'root_dir': conf_args['root_dir'],
        'lib_type': lib_type,
        'num_samples': num_samples
    }
    contents = [render('templates/merge_lanes_fastq.j2', context)]

    cell_write_dw_file = Cell(contents=contents, description="#### Merge lanes of FASTQ files")
    cells.extend(cell_write_dw_file.to_list())

    execute_cell = CellSbatch(contents=[merge_fn],
                              description="Execute file to merge lanes of FASTQ files",
                              depends_on=True,
                              array="0-%d%%20" % (num_samples-1),
                              script_output="%s_%%A_%%a.out" % inspect.stack()[0][3])
    cells.extend(execute_cell.to_list())

    return cells


def cwl_json_gen(conf_args, lib_type, metadata_filename):
    func_name = inspect.stack()[0][3]
    cells = []
    output_fn = "%s/processing/%s/scripts/%s_%s.sh" % (conf_args['root_dir'],
                                                       lib_type,
                                                       func_name,
                                                       conf_args['project_name'])
    context = {
        'output_fn' : output_fn,
        'metadata_filename': metadata_filename,
        'project_name': conf_args['project_name'],
        'root_dir': conf_args['root_dir'],
        'lib_type': lib_type,
        'star_genome': consts.star_genome,
        'mem': consts.mem[lib_type.lower()],
        'nthreads': consts.nthreads[lib_type.lower()],
        'separate_jsons': consts.separate_jsons
    }
    contents = [render('templates/%s.j2' % func_name, context)]

    cell_write_dw_file = Cell(contents=contents, description="#### Create JSON files for CWL pipeline files")
    cells.extend(cell_write_dw_file.to_list())

    execute_cell = CellSbatch(contents=[output_fn],
                              description="Execute file to ungzip FASTQ files",
                              depends_on=True,
                              script_output="%s_%%A.out" % inspect.stack()[0][3])
    cells.extend(execute_cell.to_list())
    return cells


def cwl_slurm_array_gen(conf_args, lib_type, metadata_filename, pipeline_type, n_samples):
    func_name = inspect.stack()[0][3]
    cells = []
    output_fn = "%s/processing/%s/scripts/%s-%s.sh" % (conf_args['root_dir'],
                                                       lib_type,
                                                       conf_args['project_name'],
                                                       pipeline_type)
    metadata_basename = os.path.splitext(os.path.basename(metadata_filename))[0]
    context = {
        'output_fn' : output_fn,
        'metadata_basename': metadata_basename,
        'project_name': conf_args['project_name'],
        'root_dir': conf_args['root_dir'],
        'user_duke_email': conf_args['user_duke_email'],
        'lib_type': lib_type,
        'mem': consts.mem[lib_type.lower()],
        'nthreads': consts.nthreads[lib_type.lower()],
        'pipeline_type': pipeline_type
    }
    contents = [render('templates/%s.j2' % func_name, context)]

    cell_write_dw_file = Cell(contents=contents, description="#### Create SLURM array master bash file for %s samples" % pipeline_type)
    cells.extend(cell_write_dw_file.to_list())

    execute_cell = CellSbatch(contents=[output_fn],
                              description="Execute SLURM array master file",
                              depends_on=True,
                              array="0-%d%%20" % (n_samples - 1),
                              script_output="%s_%%A_%%a.out" % inspect.stack()[0][3])
    cells.extend(execute_cell.to_list())

    return cells


def get_pipeline_types(samples_df):
    lib_type = samples_df['library type'].iloc[0].lower().replace('-', '_')
    if lib_type == consts.library_type_chip_seq:
        for seq_end in consts.seq_ends:
            for peak_type in consts.peak_types:
                for with_control in consts.with_controls:
                    samples_filter = \
                        (samples_df['paired-end or single-end'].str.lower() == seq_end) \
                        & (samples_df['peak type'].str.lower() == peak_type)
                    if with_control:
                        samples_filter = samples_filter & (~samples_df['control'].isnull())
                        pipeline_type = '-'.join([seq_end, peak_type, with_control])
                    else:
                        samples_filter = samples_filter & (samples_df['control'].isnull())
                        pipeline_type = '-'.join([seq_end, peak_type])
                    yield pipeline_type, np.sum(samples_filter)
    if lib_type == consts.library_type_rna_seq:
        for seq_end in consts.seq_ends:
            for strandness in consts.strandnesses:
                samples_filter = \
                    (samples_df['paired-end or single-end'].str.lower() == seq_end) \
                    & (samples_df['strand specificity'].str.lower() == strandness)
                if consts.with_sjdb:
                    pipeline_type = '-'.join([seq_end, strandness, 'with-sjdb'])
                else:
                    pipeline_type = '-'.join([seq_end, strandness])
                yield pipeline_type, np.sum(samples_filter)


def create_cells(samples_df, conf_args=None):
    """
    Master function to write all code and text for the notebook.

    It has a number of components:
        - write metadata txt file
        - write bash file to download FASTQ.gz files from sequencing core
        - execute previous file in HARDAC
        - write file to uncompress FASTQ files
        - execute previous file in HARDAC
        - write file to rename and move FASTQ files
        - execute cwltool master CWL generator file
    """
    lib_type = samples_df.iloc[0]['library type'].lower().replace('-', '_')
    num_samples=samples_df.shape[0]
    cells = []

    cc, metadata_file = save_metadata(samples_df, conf_args, lib_type)
    cells.extend(cc)

    cells.extend(download_fastq_files(conf_args, lib_type, metadata_filename=metadata_file))
    cells.extend(ungzip_fastq_files(conf_args, lib_type, metadata_filename=metadata_file, num_samples=num_samples))
    cells.extend(merge_fastq_files(conf_args, lib_type, metadata_filename=metadata_file, num_samples=num_samples))
    cells.extend(cwl_json_gen(conf_args, lib_type, metadata_filename=metadata_file))
    for pipeline_type, n in get_pipeline_types(samples_df):
        if n>0:
            cells.extend(cwl_slurm_array_gen(conf_args, lib_type, metadata_filename=metadata_file,
                                             pipeline_type=pipeline_type, n_samples=n))
    return cells


def make_notebook(outfile, metadata, conf_args=None):
    """Create notebook with parsed contents from metadata"""
    nb = nbf.new_notebook()

    cells = []
    # Create a notebook by Library type existing in the metadata file
    for samples_df in get_samples_by_libray_type(metadata, conf_args['sep']):
        cells.extend(create_cells(samples_df, conf_args=conf_args))

    nb['worksheets'].append(nbf.new_worksheet(cells=cells))

    with open(outfile, 'w') as _:
        nbformat.write(nb, _)


def get_samples_by_libray_type(metadata_file, sep='\t'):
    """
    Parse a metadata file (either a spreadsheet or a tab-delimited file.

    :return: generator of panda's dataframe
    """
    try:
        md = pd.read_excel(metadata_file)
    except XLRDError:
        md = pd.read_csv(metadata_file, sep=sep)

    md.columns = [x.lower() for x in md.columns]
    named_cols = [c for c in md.columns if not c.startswith('unnamed: ')]
    lib_types_found = set(md['library type'][~pd.isnull(md['library type'])])

    for lt in lib_types_found:
        yield md.loc[md['library type'] == lt, named_cols]


def main():
    parser = argparse.ArgumentParser('Generator of Jupyter notebooks to execute CWL pre-processing pipelines')
    parser.add_argument('-o', '--out', required=True, type=str, help='Jupyter notebook output file name')
    parser.add_argument('-c', '--conf-file', required=True, type=file, help='YAML configuration file (see examples)')
    parser.add_argument('-m', '--metadata', required=True, type=file, help='Metadata file with samples information')
    parser.add_argument('-f', '--force', action='store_true', help='Force to overwrite output file')
    parser.add_argument('--metadata-sep', dest='sep', required=False, type=str,
                        help='Separator for metadata file (when different than Excel spread sheet)')
    parser.add_argument('--project-name', required=False, type=str,
                        help='Project name (by default, basename of metadata file name)')

    args = parser.parse_args()

    conf_args = ruamel.yaml.load(args.conf_file, Loader=ruamel.yaml.Loader)
    if args.project_name:
        conf_args['project_name'] = args.project_name
    else:
        project_name = os.path.basename(args.metadata.name)
        conf_args['project_name'] = os.path.splitext(project_name)[0]

    outfile = "%s.ipynb" % conf_args['project_name']

    if os.path.isdir(args.out):
        outfile = os.path.join(args.out, outfile)
    else:
        outfile = args.out

    if os.path.isfile(outfile) and not args.force:
        print outfile, "is an existing file. Please use -f or --force to overwrite the contents"
        sys.exit(1)

    make_notebook(outfile,
                  args.metadata,
                  conf_args=conf_args)


if __name__ == '__main__':
    main()