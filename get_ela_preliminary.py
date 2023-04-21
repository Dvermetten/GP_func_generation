

import warnings
from preliminary.get_ela_bbob import get_ela_bbob
from preliminary.get_ela_normalize import get_ela_normalize
from preliminary.get_ela_corr import get_ela_corr
from preliminary.get_ela_dist import get_ela_dist
from preliminary.get_ela_target import get_ela_target

    
#%%
def main():
    dim = 2
    ndoe = 150*dim
    np = 8
    
    get_ela_bbob(dim, ndoe, np=np)
    get_ela_normalize(dim)
    get_ela_corr(dim)
    get_ela_dist(dim, np=np)
    get_ela_target(dim, np=np)
# END DEF

#%%
if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=RuntimeWarning) 
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
# END IF