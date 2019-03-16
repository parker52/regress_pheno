# regress_pheno

Run a multiple linear regression of one phenotype against one or more phenotypes 

### Script Overview

This script takes in a tab-delimited clinical traits file from the HMDP database, and will run a mutliple regression of one phenotype on one or more other phenotypes. The file should be preprocessed using preprocess.py from the [mouseGWASanalysis repository](https://github.com/nlapier2/mouseGWASAnalysisPackage). 

Use Python3 is required to use this script. To run the script, simply run the following command. In this example, the file is called **preprocessed_clinical_traits_chow2.tsv** and the phenotype to run the multiple linear regression on from the file is **Lean_mass**, which is being regressed against the three phenotypes **Fat_mass**, **HDL**, and **NMR_total_Mass** (which should be separated by a space).  
```
python3 regress_pheno.py --clinical /path/to/my/preprocessed_clinical_traits_chow2.tsv --target Lean_mass --regress Fat_mass HDL NMR_total_Mass
```

