import _init_paths
from datasets.factory import get_imdb

imdb = get_imdb('schalke')


print imdb._do_python_eval('/home/andras/daniel', True) 
