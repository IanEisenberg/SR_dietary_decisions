import numpy as np
from rpy2.robjects import pandas2ri, Formula
from rpy2.robjects.packages import importr
pandas2ri.activate()

def convert_to_dataframe(r_matrix):
    base = importr('base')
    df = pandas2ri.ri2py_dataframe(r_matrix)
    df.columns = base.colnames(r_matrix)
    df.index = base.rownames(r_matrix)
    return df

def polr(data, formula):
    "Python wrapper around the MASS function polr in R"
    # reference: https://www.analyticsvidhya.com/blog/2016/02/multinomial-ordinal-logistic-regression/
    base = importr('base')
    stats = importr('stats')
    MASS = importr('MASS')
    out = MASS.polr(Formula(formula), data)
    summary = base.summary(out)
    coefs = convert_to_dataframe(summary[0])
    # confidence intervals
    confint = convert_to_dataframe(stats.confint(out))
    # get pvalue
    pvals = np.array(stats.pnorm(abs(coefs['t value']), **{"lower.tail": False}))**2
    coefs.loc[:, 'p value'] = pvals
    coefs = coefs.join(confint)
    return {'polr_out': out,
            'summary': summary,
            'coefs': coefs}
    

