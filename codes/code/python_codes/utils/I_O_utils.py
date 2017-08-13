def mkdir_p(mypath):
    """
    Creates a directory. equivalent to using mkdir -p on the command line
    
    @param mypath - local path 
    
    --------------
    Change the root_path in accordance to your wished root path
    """

    from errno import EEXIST
    from os import makedirs,path
    import os

    root_path = ''
    
    try:
        makedirs(root_path + mypath)
    except OSError as exc:
        if exc.errno == EEXIST and path.isdir(root_path + mypath):
            pass
        else: raise
