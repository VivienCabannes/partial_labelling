
import pathlib

# Paths to be change by user
data_dir = pathlib.Path.home() / 'data'
LIBSVM_DIR = data_dir / 'classification'
MULAN_DIR = data_dir / 'multilabel'

# Metadata, if using Mulan, it is necessary to specify the right path
MULAN_INFO = MULAN_DIR / 'info.json'
LIBSVM_INFO = LIBSVM_DIR / 'info.json'
