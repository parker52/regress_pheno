"""
This script runs a linear regression model on specified phenotypes given an input 
data file 
"""

import argparse, math, sys, numpy as np, scipy as sp, seaborn as sns, matplotlib.pyplot as plt
import pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
import scipy.stats as stats
from sklearn import preprocessing

def parseargs():    # handle user arguments
  parser = argparse.ArgumentParser(description="Given phenotype file," +
        " get into proper format for pylmm.")
  parser.add_argument('--clinical', required = True,
    help = 'Name of existing pheno file.')
  parser.add_argument('--target', required=True,
    help = 'EXACT name of target trait to study.')
  parser.add_argument('--output', default='regression_summary.txt',
    help = 'Name of output file.')
  parser.add_argument('--regress', default = 'Fat_mass', nargs='+',
    help = 'Use this to regress target phenotype on other phenotypes.')
  parser.add_argument('--regress_type', default='multiple',
    help = 'Specify multiple regression or pairwise regression.')
  args = parser.parse_args()
  return args

def parse_clinical_file(args):
  """
  Given a target phenotype, read the values for that phenotype.
  Arguments:
  -- args are the user arguments parsed with argparse
  Returns: 
  -- target_pheno: just the raw target phenotype values
  -- other_phenos: just the raw non-target (other) phenotype values
  """
  # read in clinical file  
  target_pheno, other_phenos = [], []
  with(open(args.clinical, 'r')) as infile:
    header = infile.readline().strip().split('\t')
    targetcol = header.index(args.target)
    mousecol, straincol = header.index('mouse_number'), header.index('Strain')
    for line in infile:
      splits = line.strip().split('\t')
      if len(splits) < 2:
        break
      tgt = splits[targetcol]
      if tgt != 'NA':
        tgt = float(tgt)
      if tgt == 'NA':
        tgt = np.NaN
      others = [splits[i] for i in range(len(splits))
        if i not in [straincol, mousecol, targetcol]]
      target_pheno.append(tgt)
      other_phenos.append(others)

  return target_pheno, other_phenos, header

def format_data(args, target_pheno, other_phenos, header):
  """ 
  Format the data, remove all individuals for which there are NaN indices for any of the phenotypes 
  Arguments:
  -- args: the user arguments passed with argparse 
  -- target_pheno: just the raw target phenotype values returned from parse_clinical_files 
  -- other_phenos: just the raw non-target (other) phenotype values returned from parse_clinical_files 
  -- header: list of all the phenotype headers, stripped and split
  Returns:
  -- copy_target: the data for the target phenotype, with individuals removed for which there is 
                  a NaN value for any of the phenotypes
  -- copy_regress: a list of lists of the data for the regression phenotypes, with individuals 
                   removed for where there is a NaN value for any of the phenotypes
  """
  targetcol = header.index(args.target)
  mousecol, straincol = header.index('mouse_number'), header.index('Strain')
  other_cols = [header[i] for i in range(len(header))
    if i not in [targetcol, mousecol, straincol]]   
  # create index of regress phenotype (just phenotype name)
  regresscols = [header.index(i) for i in args.regress]
  # create list other_pheno_data, each element is a list of the data entries for a single phenotype (column)
  other_pheno_data = []
  for i in range(len(other_cols)):
    other_pheno_data.append([row[i] for row in other_phenos])
  # convert other_pheno_data to numpy array
  other_pheno_data = np.array(other_pheno_data)
  # check for NA values and convert to NaN for all phenotypes
  for i in range(len(other_phenos)):
    for entry in other_pheno_data:
      if entry[i] == 'NA':
        entry[i] = np.NaN
  # create dictionary for other_cols: other_pheno_data
  other_dict = dict(zip(other_cols, other_pheno_data))
  # need to code sex as M = 0, F = 1
  for n, i in enumerate(other_dict['Sex']):
    if i == 'M':
      other_dict['Sex'][n] = '0'
    if i == 'F':
      other_dict['Sex'][n] = '1'
  # create dictionary for targetcol: target_pheno
  target_data = {}
  target_data[targetcol] = target_pheno
  # convert all non nan values to floats for target_pheno
  for key in target_data.keys():
    target_data[key] = [np.float(i) if i != 'nan' else np.nan for i in target_data[key]]
  # target_nan_indices = list of indices on nan values for target_pheno
  target_nan_indices = []
  for i in range(len(target_pheno)):
    if np.isnan(target_pheno[i]):
      target_nan_indices.append(i)
  # convert all non nan values to floats for all phenotypes 
  for key in other_dict.keys():
    other_dict[key] = [np.float(i) if i != 'nan' else np.nan for i in other_dict[key]]
  # other_nan_indices = list of lists conatining the indices of nan values for other_phenos
  other_nan_indices = []
  for pheno in other_dict.values():
    row_indices = []
    for i in range(len(other_phenos)):
      if np.isnan(pheno[i]):
        row_indices.append(i)
    other_nan_indices.append(row_indices)
  # target_nan_indices = list of indices of nan values for target_pheno
  target_nan_indices = []
  for i in range(len(target_pheno)):
    if np.isnan(target_pheno[i]):
      target_nan_indices.append(i)
  # create dictionary: {phenotype (header) : [indices of rows with nan values for that phenotype]}
  other_nan_index_dict = dict(zip(other_cols, other_nan_indices))
  # create copies of the two traits (--target and --regress)
  copy_target = target_pheno
  copy_regress = [other_dict[args.regress[i]] for i in range(len(args.regress))]
  # make list of all indices that have nan values for the phenotypes to be regressed on
  regress_nan_indices = []
  for key in other_nan_index_dict.keys():
    if key in args.regress:
      for index in other_nan_index_dict[key]:
        if index not in regress_nan_indices:
          regress_nan_indices.append(index)
  regress_nan_indices.sort()
   
  joint_indices = np.unique(target_nan_indices + regress_nan_indices)
  reversed_joint_indices = joint_indices[::-1]
  # delete nan values from copy_target phenotype data
  for index in reversed_joint_indices:
    del copy_target[index]
  # delete nan values from copy_regress phenotypes data
  for i in range(len(copy_regress)):
    for index in reversed_joint_indices:
      del copy_regress[i][index]

  return copy_target, copy_regress

def run_model(args, copy_target, copy_regress):
  """
  Run multiple linear regression model 
  Arguments:
  -- args: the user arguments passed with argparse
  -- copy_target: the data for the target phenotype, with individuals removed for which there is 
		  a NaN value for any of the phenotypes
  -- copy_regress: a list of lists of the data for the regression phenotypes, with individuals 
		   removed for where there is a NaN value for any of the phenotypes 
  Returns: 
  -- model: model object from statsmodels linear regression
  -- data: pre-processed pandas dataframe that contains both target and regression phenotypes 
  """
  # all_phenos = a dataframe-like object for input to residplot
  all_phenos = [copy_target] + copy_regress
  pheno_headers = [args.target] + args.regress # create list of target_pheno header and regress_pheno headers
  data = pd.DataFrame(all_phenos)
  data = data.transpose()
  data.columns = pheno_headers
  # Drop the 127th value for testing (outlier)
  # data = data.drop([127])
  # print(tabulate(data, headers='keys', tablefmt='psql'))
  X = data[args.regress]
  y = data[args.target]
  X = sm.add_constant(X)
  model = sm.OLS(y, X).fit()
  # print(model.summary())
 
  return model, data


def plot_model(args, model, data):
  """
  Write OLS Model Summary to 'regression_summary.txt'
  Plot the Residuals vs Fitted, Normal Q-Q, Scale-Location, and Residuals vs Leverage plots
  Arguments:
  -- model: model object from statsmodels linear regression
  -- data: pre-processed pandas dataframe that contains both target and regression phenotypes 
  """
  # write OLS model summary to 'regression_summary.txt'
  summary = model.summary()
  f = open('regression_summary.txt', 'w')
  f.write(str(summary))
  f.close() 
  print(model.summary())

  # creating residual plots 
  model_fitted_y = model.fittedvalues
  model_residuals = model.resid
  model_norm_residuals = model.get_influence().resid_studentized_internal # normalized model residuals
  model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals)) # absolute square root normalized residuals
  model_abs_resid = np.abs(model_residuals) # absolute model residuals
  model_leverage = model.get_influence().hat_matrix_diag
  model_cooks = model.get_influence().cooks_distance[0]

  plt.figure(num=None, figsize=(16,16), dpi=80, facecolor='w',edgecolor='k')
  ax = plt.subplot(2,2,1)
  sns.residplot(model_fitted_y, args.target, data=data,
      lowess=True,
      scatter_kws={'alpha': 0.5},
      line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
  ax.set_title('Residuals vs Fitted')
  ax.set_xlabel('Fitted values')
  ax.set_ylabel('Residuals')
  ax = plt.subplot(2,2,2)
  stats.probplot(model_residuals, dist="norm", plot=plt)
  plt.title("Normal Q-Q plot")
  ax.set_title('Normal Q-Q')
  ax.set_xlabel('Theoretical Quantiles')
  ax.set_ylabel('Residuals');
  ax = plt.subplot(2,2,3)  
  plt.scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5)
  sns.regplot(model_fitted_y, model_norm_residuals_abs_sqrt,
            scatter=False,
            ci=False,
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
  ax.set_title('Scale-Location')
  ax.set_xlabel('Fitted values')
  ax.set_ylabel('$\sqrt{|Standardized Residuals|}$');
  ax = plt.subplot(2,2,4)
  plt.scatter(model_leverage, model_norm_residuals, alpha=0.5)
  sns.regplot(model_leverage, model_norm_residuals,
            scatter=False,
            ci=False,
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
  ax.set_xlim(0, 0.20)
  ax.set_ylim(-3, 5)
  ax.set_title('Residuals vs Leverage')
  ax.set_xlabel('Leverage')
  ax.set_ylabel('Standardized Residuals')

  def graph(formula, x_range, label=None):
    x = x_range
    y = formula(x)
    plt.plot(x, y, label=label, lw=1, ls='--', color='red')

  p = len(model.params) # number of model parameters

  graph(lambda x: np.sqrt((0.5 * p * (1 - x)) / x),
      np.linspace(0.001, 0.200, 50),
      'Cook\'s distance') # 0.5 line
  graph(lambda x: np.sqrt((1 * p * (1 - x)) / x),
      np.linspace(0.001, 0.200, 50)) # 1 line
  plt.legend(loc='upper right');

  plt.savefig('model_plots.png')

  return 

# def pair_regress_pheno(args, target_pheno, other_phenos, header):
#   return
# for later implementation

def main():
  args = parseargs()  # handle user arguments
  # read the phenotype values from the clinical trait file
  target_pheno, other_phenos, header = parse_clinical_file(args)
  # if args.regress_type == 'pair':
  #   pair_regress_pheno(args, target_pheno, other_phenos, header)
  copy_target, copy_regress = format_data(args, target_pheno, other_phenos, header)
  model, data = run_model(args, copy_target, copy_regress)
  plot_model(args, model, data)

if __name__ == '__main__':
  main()

