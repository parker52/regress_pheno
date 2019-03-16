# regress_pheno

Run a multiple linear regression of one phenotype against one or more phenotypes 

### Script Overview

This script takes in a tab-delimited clinical traits file from the HMDP database, runs a mutliple regression of one phenotype on one or more other phenotypes, and writes the regression summary of results to a text file and the regression plots to a png file. The file should be preprocessed using preprocess.py from the [mouseGWASAnalysisPackage](https://github.com/nlapier2/mouseGWASAnalysisPackage). 

### Running the Script

Use Python3 is required to use this script. To run the script, simply run the following command. In this example, the file is called **preprocessed_clinical_traits_chow2.tsv** and the dependent variable from the file is **Lean_mass**, which is being regressed against the three phenotypes **Fat_mass**, **HDL**, and **NMR_total_Mass** (which should be separated by a space in the command line).  
```
python3 regress_pheno.py --clinical /path/to/my/preprocessed_clinical_traits_chow2.tsv --target Lean_mass --regress Fat_mass HDL NMR_total_Mass
```

The script will write the [summary of OLS](https://www.statsmodels.org/dev/examples/notebooks/generated/ols.html) results from the mutliple linear regression to a text file  **regression_summary.txt** to the current directory. 

The script will save the Residual Plot, QQ Plot, Scale-Location Plot, and Leverage Plot to a png file **model_plots.png** to the current directory. 
