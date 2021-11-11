#Q3
from scipy.stats import norm
from scipy.stats import geom
import  math

def Q3_a():
    print("the probabilty that one plane will weigh more than 82 kgs is " +str( 1-norm.cdf(82,loc=81.5,scale=13)))
def Q3_b():
    print(str((1 - norm.cdf(82, loc=81.5, scale=13))**50))
def Q3_c():
   print(str(1-norm.cdf(82,loc=81.5,scale=13/math.sqrt(50))))
def Q3_d():
    print(norm.cdf((82-81.5)/(13/5),loc=0,scale=1)-norm.cdf((80-81.5)/(13/5),loc=0,scale=1))
def Q3_e():
   print(norm.ppf(0.9,loc=81.5,scale=13))
def Q3_f():
    sigma = 20 / (norm.ppf(0.7, loc=0, scale=1) - norm.ppf(0.2, loc=0, scale=1))
    E = 70 - sigma * norm.ppf(0.2, loc=0, scale=1)
    print("std")
if __name__ == '__main__':
    Q3_a()
    Q3_b()
    Q3_c()
    Q3_d()
    Q3_e()
    Q3_f()