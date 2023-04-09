import numpy as np
import pandas as pd
import statistics as stat
import scipy.optimize as sc

class Value:
    var: float
    err: float

    @property
    def absolute_error(self):
        return self.err

    @property
    def r_err(self):
        if abs(self.var) < 0.00000001:
            return 0.0
        else:
            return abs(self.err / self.var)

    def __init__(self, var, err=0.0, r_err=None):
        self.var = float(var)
        if r_err is None:
            self.err = abs(err)
        else:
            self.err = abs(var * r_err)


    def __str__(self):
        r = 1
        while round(self.err, r) == 0 and r < 30:
            r += 1
        a = '{:6f}'.format(round(self.var, r))
        while a[-1] == "0":
            a = a[:-1]
        b = '{:6f}'.format(round(self.err, r))
        while b[-1] == "0":
            b = b[:-1]
        return f"({a}\u00B1{b})"
    
    def __repr__(self):
        return str(self)
    
    
    def __int__(self):
        return int(self.var)

    def __float__(self):
        return self.var


    def __pos__(self):
        return self

    def __neg__(self):
        return Value(-self.var, self.err)

    def __abs__(self):
        return Value(abs(self.var), self.err)

    def __round__(self, ndigits=None):
        return Value(round(self.var, ndigits), round(self.err, ndigits))


    def __add__(self, other):
        if isinstance(other, Value):
            return Value(self.var + other.var, (self.err ** 2 + other.err ** 2) ** 0.5)
        elif isinstance(other, int) or isinstance(other, float):
            return Value(self.var + other, self.err)
        else:
            raise TypeError

    def __radd__(self, other):
        if isinstance(other, Value):
            return Value(self.var + other.var, (self.err ** 2 + other.err ** 2) ** 0.5)
        if isinstance(other, int) or isinstance(other, float):
            return Value(self.var + other, self.err)
        else:
            raise TypeError
    
    def __iadd__(self, other):
        self = self + other


    def __sub__(self, other):
        if isinstance(other, Value):
            return Value(self.var - other.var, (self.err ** 2 + other.err ** 2) ** 0.5)
        elif isinstance(other, int) or isinstance(other, float):
            return Value(self.var - other, self.err)
        else:
            raise TypeError
    
    def __rsub__(self, other):
        if isinstance(other, Value):
            return Value(self.var - other.var, (self.err ** 2 + other.err ** 2) ** 0.5)
        elif isinstance(other, int) or isinstance(other, float):
            return Value(other - self.var, self.err)
        else:
            raise TypeError
    
    def __isub__(self, other):
        self = self - other
    

    def __mul__(self, other):
        if isinstance(other, Value):
            return Value(self.var * other.var, r_err=(self.r_err ** 2 + other.r_err ** 2) ** 0.5)
        elif isinstance(other, int) or isinstance(other, float):
            return Value(self.var * other, r_err=self.r_err)
        else:
            raise TypeError
    
    def __rmul__(self, other):
        if isinstance(other, Value):
            return Value(self.var * other.var, r_err=(self.r_err ** 2 + other.r_err ** 2) ** 0.5)
        elif isinstance(other, int) or isinstance(other, float):
            return Value(self.var * other, r_err=self.r_err)
        else:
            raise TypeError
    
    def __imul__(self, other):
        self = self * other


    def __truediv__(self, other):
        if isinstance(other, Value):
            return Value(self.var / other.var, r_err=(self.r_err ** 2 + other.r_err ** 2) ** 0.5)
        elif isinstance(other, int) or isinstance(other, float):
            return Value(self.var / other, r_err=self.r_err)
        else:
            raise TypeError
    
    def __rtruediv__(self, other):
        if isinstance(other, Value):
            return Value(self.var / other.var, r_err=(self.r_err ** 2 + other.r_err ** 2) ** 0.5)
        if isinstance(other, int) or isinstance(other, float):
            return Value(other / self.var, r_err=self.r_err)
        else:
            raise TypeError

    def __itruediv__(self, other):
        self = self / other
  

    def __pow__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Value(self.var ** other, self.var ** (other - 1) * other * self.err)
        else:
            raise TypeError
    

    def __lt__(self, other):
        if isinstance(other, Value):
            return self.var < other.var
        elif isinstance(other, int) or isinstance(other, float):
            return self.var < other
        else:
            raise TypeError
        
    def __gt__(self, other):
        if isinstance(other, Value):
            return self.var > other.var
        elif isinstance(other, int) or isinstance(other, float):
            return self.var > other
        else:
            raise TypeError
    

    def __hash__(self):
        return hash(self.var) * int("1" + "0" * int(len(str(hash(self.var))))) + hash(self.err)

def series_err(var):
    def f(var):
        return Value(var, next(f.it))

    f.it = iter(var)
    return f

def const_err(err):
    def f(var):
        return Value(var, err)

    return f

def err(x):
    if isinstance(x, Value):
        return x.err
    elif isinstance(x, int) or isinstance(x, float):
        return float(0)
    else:
        raise TypeError

def get_err(x: Value):
    if isinstance(x, Value):
        return x.err
    elif isinstance(x, int) or isinstance(x, float):
        return float(0)
    else:
        raise TypeError

def get_var(x: Value):
    if isinstance(x, Value):
        return x.var
    elif isinstance(x, int) or isinstance(x, float):
        return float(x)
    else:
        raise TypeError()

def value_from_series(s):
    if len(s) <= 1:
        raise IndexError
    return Value(np.mean(s), np.sqrt(np.sum((s - np.mean(s)) ** 2) / len(s) / (len(s) - 1)))

def mean(s):
    return stat.mean(s)

def line(x, a, b):
    return get_var(a) * x.agg(get_var) + get_var(b)

def curve_fit(xdata, ydata, f=line):
    params, cov = sc.curve_fit(f, xdata=xdata.agg(get_var), ydata=ydata.agg(get_var))

    return (Value(params[i], np.sqrt(cov[i][i])) for i in range(len(params)))