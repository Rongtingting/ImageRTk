"""ImageRTk: A toolkit for image-based spatial analysis and modeling.
"""

from .version import __version__
from time import gmtime, strftime
print (f'(Running ImageRTk {__version__})')
print (strftime("%Y-%m-%d %H:%M:%S", gmtime()))

# set simplified alias
from . import preprocessing as pp
from . import plot as pl
from . import analysis as al
from . import model
from . import data
from . import utils
from . import io
from . import segmentation as seg