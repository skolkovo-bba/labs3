import numpy as np
import pandas as pd
import statistics as stat
import scipy.optimize as sc
import matplotlib.pyplot as plt



class Value:
    var: float
    err: float

    @property
    def absolute_error(self):
        return self.err

    @property
    def r_err(self):
        if abs(self.var) < 0.00000000000000000001:
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
        # Определение e
        if np.log10(np.abs(self.var)) < 0:
            ten = int(np.ceil(np.abs(np.log10(np.abs(self.var)))) * np.sign(np.log10(np.abs(self.var))))
        else:
            ten = int(np.floor(np.abs(np.log10(np.abs(self.var)))))

        # Определение полезную точность
        r = 0
        while str(round(self.err * 10 ** (-ten), r))[-1] == "0" and r < 4:
            r += 1
        
        # В любом случае округляем до полезной точности
        a = round(self.var * 10 ** (-ten), r)
        b = round(self.err * 10 ** (-ten), r)

        if -2 < ten <= 2:
            return f"({round(a * 10 ** ten, 4)}\u00B1{round(b * 10 ** ten, 4)})" # Дополнительное округление на случай фантомных знаков при умножение на 10 в степени
        else:
            return f"({a}\u00B1{b}e{ten})"
            

    
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
            raise TypeError(f"{type(self)} and {type(other)}")
    
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
            raise TypeError(f"{type(self)} and {type(other)}")
        
    def __gt__(self, other):
        if isinstance(other, Value):
            return self.var > other.var
        elif isinstance(other, int) or isinstance(other, float):
            return self.var > other
        else:
            raise TypeError
    

    def __hash__(self):
        return hash(self.var) * int("1" + "0" * int(len(str(hash(self.var))))) + hash(self.err)

q_e = 1.6021766208 * 10 ** -19
pi = np.pi
g = Value(9.81, 0.01)

def series_err(var):
    def f(var):
        return Value(var, next(f.it))

    f.it = iter(var)
    return f

def const_err(err):
    def f(var):
        return Value(var, err)

    return f

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

err = get_err
var = get_var

def value_from_series(s):
    if len(s) <= 1:
        raise IndexError
    return Value(np.mean(s), np.sqrt(np.sum((s - np.mean(s)) ** 2) / len(s) / (len(s) - 1)))

def mean(s):
    return stat.mean(s.agg(var))

def line(x, a, b):
    return var(a) * x.transform(var) + var(b)



def mls(f, x, y):
    if isinstance(x, pd.Series):
        x = x.transform(var)
    if isinstance(y, pd.Series):
        y = y.transform(var)
    params, cov = sc.curve_fit(f, xdata=x, ydata=y)

    return (Value(params[i], np.sqrt(cov[i][i])) for i in range(len(params)))

def hi2(x, y):
    x = x.agg(var)
    y = y.agg(var)
    def mean_w(x):
        sum((1 / (v.err ** 2) for v in x)) / sum((1 / (v.err ** 2) for v in x))
    mean_w.x = mean_w

    print(x, mean_w(x ** 2), mean_w(x) ** 2)

    k = (mean_w(x * y) - mean_w(x) * mean_w(y)) / (mean_w(x ** 2) - mean_w(x) ** 2)
    b = mean_w(y) - k * mean_w(x)

    return k, b

def errorbar(x, y, title, xlabel, ylabel, fontsize=25):
    plt.figure(figsize=(16, 9), dpi=254)

    k, b = mls(line, x, y)

    plt.errorbar(x=x, y=y, xerr=x.transform(get_err), yerr=y.transform(get_err), fmt='ko')
    plt.plot(x, line(x, k, b), color="black")

    plt.title(title, fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.grid(True)

    return k, b

def read_csv(name):
    pass

#TODO Проверка работы со всему разными функциями из numpy - Вроде все робит
#TODO Функции для замены matplotlib - Стоит допилить функицонал для нескольких графиков на одной плоскости
#TODO Правильный вывод чисел - Походу сделано
#TODO Read_csv с возможностью читать погрешности