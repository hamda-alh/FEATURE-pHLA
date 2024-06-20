# FEATURE-pHLA
## Physio-chemical features is all you need!
### Overview
We present FEATURE-pHLA. A tree-based model utilizing the physico-chemical fingerprints obtained from peptides and HLA sequences to predict pHLA binding affinity. To determine the predictive capabilities in physio-chemical features and explaining the mechanisms underlying the binding process. Below is the Framework of FEATURE-pHLA.



![abstract_summary drawio](https://github.com/hamda-alh/FEATURE-pHLA/assets/152274710/f8d987fb-70c9-4788-81d8-20ac0ae93ed0)
## 

## Data:
Source data is acquired from (https://data.mendeley.com/datasets/zx3kjzc3yx/3). Ensure to download the data to the Data folder.

## Requirements
* run pip install -r requirements.txt
* requires CPU only

## FEATURE-pHLA
 * To extract the list of physio-chemical features of the peptide and HLA sequences run [pHLA_data.ipynb](Script/pHLA_data.ipynb)
 * To retrain the model run [feature_based_lgbm_pred.ipynb](Script/feature_based_lgbm_pred.ipynb)
 * To generate a SHAPley based feature importance plot run [SHAP_figure_lightgbm.ipynb](Script/SHAP_figure_lightgbm.ipynb)

