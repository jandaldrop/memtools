class flist(list):
    """
Listlike object, where the getitem operation is
replaced by a function call.
I.e. flist can store a list of filenames, which
are loaded into memory only on access.
The number of function calls are counted in flist.n_calls
    """
    def __init__(self, load_function, r=list()):
        super(flist, self).__init__(r)
        self.f=load_function
        self.n_calls=0
    def __iter__(self):
        for x in list.__iter__(self):
            self.n_calls+=1
            yield self.f(x)
    def __getitem__(self, n):
        self.n_calls+=1
        return self.f(super(flist, self).__getitem__(n))
