class flist(list):
    """
Listlike object, where the getitem operation is
replaced by a function call.
I.e. flist can store a list of filenames, which
are loaded into memory only on access.
The result of the last function call is stored.
The number of function calls are counted in flist.n_calls
    """
    def __init__(self, f_handle, r=list()):
        super(flist, self).__init__(r)
        self.f_handle=f_handle
        self.n_calls=0
        self.last_arg=None
    def __iter__(self):
        for x in list.__iter__(self):
            yield self.f(x)
    def __getitem__(self, n):
        return self.f(super(flist, self).__getitem__(n))
    def f(self, x):
        if self.last_arg!=x:
            self.last_res=self.f_handle(x)
            self.last_arg=x
            self.n_calls+=1
        return self.last_res
