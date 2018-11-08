import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from r_utils import polr

dietary_data = pd.read_csv('Data/dietary_decision.csv.gz', index_col=0)
eating_data = pd.read_csv('Data/eating_survey_DVs.csv', index_col=0)

# ****************************************************************************
# decision modeling
# ****************************************************************************
decision_data = dietary_data.query('exp_stage == "decision"')
# join with eating data
decision_data = decision_data.join(eating_data, on='worker_id')
# reduce to modeling dataset
decision_data = decision_data.loc[:, ['health_diff', 
                                      'taste_diff', 
                                      'coded_response',
                                      'eating_survey.cognitive_restraint',
                                      'worker_id']]
# clean up data
# remove NaN
decision_data = decision_data.loc[~decision_data.health_diff.isnull()]
# convert from float to int
decision_data.loc[:, ['health_diff', 'taste_diff']] = \
    decision_data[['health_diff', 'taste_diff']].astype(int)
# set coded response as ordered factor
decision_data.coded_response = decision_data.coded_response.astype('category')
# rename
decision_data.rename({'eating_survey.cognitive_restraint': 'cognitive_restraint'},
                     axis=1, inplace=True)
formula = 'coded_response ~ health_diff*cognitive_restraint + taste_diff*cognitive_restraint'
# run model
out = polr(decision_data, formula)
summary = out['summary']
coefs = out['coefs']



# *****************************************************************************
# visualization
# *****************************************************************************

# get groups
restraint_data = eating_data['eating_survey.cognitive_restraint']
high_group = list(restraint_data[restraint_data>restraint_data.median()].index)
low_group = list(restraint_data[restraint_data<restraint_data.median()].index)

# mean responses
mean_responses = dietary_data.groupby(['health_diff','taste_diff']) \
        .coded_response.mean().reset_index()
full_pivoted = mean_responses.pivot(index='health_diff', 
                    columns='taste_diff', 
                    values='coded_response').iloc[::-1,:]

# high contraint
mean_responses = dietary_data.query('worker_id in %s' % high_group) \
        .groupby(['health_diff','taste_diff']) \
        .coded_response.mean().reset_index()
high_pivoted = mean_responses.pivot(index='health_diff', 
                    columns='taste_diff', 
                    values='coded_response').iloc[::-1,:]

# low contraint
mean_responses = dietary_data.query('worker_id in %s' % low_group) \
        .groupby(['health_diff','taste_diff']) \
        .coded_response.mean().reset_index()
low_pivoted = mean_responses.pivot(index='health_diff', 
                    columns='taste_diff', 
                    values='coded_response').iloc[::-1,:]

# plotting
def plot_healthtaste(data, ax=None, title=None, kwargs=None):
    """ Plots a health x taste x response matrix
    
    Args:
        data: a health_diff x taste_diff matrix of average responses
        ax (optional): axis to use for plotting
        title (optional): title to include
        kwargs (optional): kwargs to pass to seaborn's heatmap function
    """
    if ax is None:
        f, ax = plt.subplots(1, 1)
    default_kwargs = {'cbar': False}
    if kwargs is not None:
        default_kwargs.update(kwargs)
    sns.heatmap(data, square=True, vmin=-2, vmax=2,
                ax=ax, **default_kwargs)
    ax.tick_params(labelsize=10)
    ax.set_xlabel('Taste Difference', fontsize=16)
    ax.set_ylabel('Health Difference', fontsize=16)
    if title:
        ax.set_title(title, fontsize=20)
    
f, axes = plt.subplots(2,2, figsize=(14,14))
axes = f.get_axes()
plot_datasets = [(full_pivoted, 'Full Dataset', None), 
                 (high_pivoted-low_pivoted, 'High-Low', {'annot': True}),
                 (high_pivoted, 'High Constraint', None), 
                 (low_pivoted, 'Low Constraint', {'cbar': True})]

for ax, (data, title, kwargs) in zip(axes, plot_datasets):
    plot_healthtaste(data, ax, title, kwargs)
    
#f.savefig('/home/ian/choice_heatmaps.pdf')
