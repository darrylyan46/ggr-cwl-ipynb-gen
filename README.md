# ggr-cwl-ipynb-gen
Jupyter notebook generator to download and execute the processing files for GGR related datasets. 
At this point, is not intented to cover all use cases, but to serve as a quick generator of all 
related files and scripts to pre-process sequences generated at the [Duke-GCB Sequencing Core](https://genome.duke.edu/cores-and-services/sequencing-and-genomic-technologies) in [HARDAC](https://genome.duke.edu/cores-and-services/computational-solutions/compute-environments-genomics).

Example of usage:
```
$ python ggr-cwl-ipynb-gen.py \
  --conf examples/conf.yaml \
  --metadata examples/Hong_3979_170316B1.xlsx \
  --out /path/to/output_dir \
  --force
```
The information in the example metadata and configuration file should reveal what is needed to download and pre-process the samples.

For a full list of options:
```
$ python ggr-cwl-ipynb-gen.py -h
usage: Generator of Jupyter notebooks to execute CWL pre-processing pipelines
       [-h] -o OUT -c CONF_FILE -m METADATA [-f] [--metadata-sep SEP]
       [--project-name PROJECT_NAME]

optional arguments:
  -h, --help            show this help message and exit
  -o OUT, --out OUT     Jupyter notebook output file name
  -c CONF_FILE, --conf-file CONF_FILE
                        YAML configuration file (see examples)
  -m METADATA, --metadata METADATA
                        Metadata file with samples information
  -f, --force           Force to overwrite output file
  --metadata-sep SEP    Separator for metadata file (when different than Excel
                        spread sheet)
  --project-name PROJECT_NAME
                        Project name (by default, basename of metadata file
                        name)
```

### Dependencies
- jinja2 >=2.8
- nbformat >=4.0.1
- numpy >=1.10.4
- pandas >=0.17.1
- xlrd >=1.0.0
- ruamel >=0.11.11