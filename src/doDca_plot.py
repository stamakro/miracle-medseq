import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

plt.close('all')

parser = argparse.ArgumentParser(prog='doDca.py', description='')

parser.add_argument('--outcome', dest='surv', metavar='outcome', help='OS or RFS', default='OS', type=str)
parser.add_argument('--years', dest='years', metavar='YEARS', help='timepoint to evaluate', default=3.0, type=float)
parser.add_argument('--netbenefit', dest='nb', metavar='NB or NIA', help='nb if true, else nia', default=True, type=bool)
parser.add_argument('--lowessFrac', dest='lowessFrac', metavar='LOWESS', help='lowess fraction of points', default=0.5, type=float)
parser.add_argument('--postop', dest='postop', metavar='POSTOP', help='whether to include postoperative', default=0, type=int)
args = parser.parse_args()

if args.postop:
    prefix = 'postop'
else:
    prefix = 'preop'

dca = pd.read_csv('../results/%s_dca_%s_at_%.1f_years.csv' % (prefix, args.surv, args.years), index_col=0)

lowess = sm.nonparametric.lowess

if args.nb:
    y_col = 'net_benefit'
else:
    y_col = 'net_intervention_avoided'


fig,ax = plt.subplots(1,1)

colors = ['C0', 'k', 'k']
styles = ['-', '-', '--']

for i,model in enumerate(dca['model'].unique()):
    print(model)


    x = dca[dca['model'] == model]
    if model.lower() not in ['all', 'none']:
        smoothed_data = lowess(x[y_col], x['threshold'], frac=args.lowessFrac)
        ax.plot(smoothed_data[:,0], smoothed_data[:,1], color=colors[i], linestyle=styles[i], label='model')

    else:
        # smoothed_data = x[y_col]

        ax.plot(x['threshold'], x[y_col], color=colors[i], linestyle=styles[i], label=model)


ax.set_ylim(-0.05,0.6)
ax.set_xlim(0.01,0.51)
ax.legend()
ax.set_xlabel('Threshold probability', fontsize=15)
ax.set_ylabel('Net Benefit', fontsize=16)

ax.grid()
plt.show()
