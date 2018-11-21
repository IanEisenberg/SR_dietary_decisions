import rpy2
from rpy2.robjects import pandas2ri, Formula
from rpy2.robjects.packages import importr
pandas2ri.activate()

def convert_to_dataframe(r_matrix):
    base = importr('base')
    df = pandas2ri.ri2py_dataframe(r_matrix)
    df.columns = base.colnames(r_matrix)
    df.index = base.rownames(r_matrix)
    return df

def clm(data, formula):
    "Python wrapper around the MASS function polr in R"
    # reference: https://www.analyticsvidhya.com/blog/2016/02/multinomial-ordinal-logistic-regression/
    base = importr('base')
    stats = importr('stats')
    ordinal = importr('ordinal')
    out = ordinal.clm(Formula(formula), data=data)
    summary = base.summary(out)
    coefs = convert_to_dataframe(summary[4])
    # confidence intervals
    try:
        confint = convert_to_dataframe(stats.confint(out))
        coefs = coefs.join(confint)
    except rpy2.rinterface.RRuntimeError:
        pass
    return {'ordinal_out': out,
            'summary': summary,
            'coefs': coefs}
    

