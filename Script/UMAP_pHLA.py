##UMAP pHLA

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import umap
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

print("Reading Data")
data = pd.read_csv('/home/Hamda.Alhosani/stratified_sample.csv')



import plotly.express as px
data = data.drop('Unnamed: 0', axis=1)

all_numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
excluded_cols = ['allele_id', 'peptide_id']
numerical_features = [col for col in all_numerical_cols if col not in excluded_cols]
print("Identified numerical features.")

all_categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
excluded_cols = ['allele', 'hla_sequence', 'peptide']
categorical_features = [col for col in all_categorical_cols if col not in excluded_cols]
print("Identified categorical features")

# Setting up preprocessing steps
numerical_imputer = SimpleImputer(strategy='median')
categorical_imputer = SimpleImputer(strategy='most_frequent')
categorical_encoder = OneHotEncoder(handle_unknown='ignore')
numerical_scaler = StandardScaler()



print("Setting up preprocessing steps...")
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', numerical_imputer), ('scaler', numerical_scaler)]), numerical_features),
        ('cat', Pipeline([('imputer', categorical_imputer), ('encoder', categorical_encoder)]), categorical_features)
    ])


pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
print("Pipeline Created.")
data_processed = pipeline.fit_transform(data)
print("Data processed")

#UMAP model
print("Applying UMAP...")
umap_model = umap.UMAP(n_neighbors=150, min_dist=0.8, n_components=2, random_state=42)
umap_results = umap_model.fit_transform(data_processed)



# Plotting
print("Plotting UMAP projections..")

classes = data['Locus']
plt.figure(figsize=(12, 10))
sns.scatterplot(x=umap_results[:, 0], y=umap_results[:, 1], hue=classes, palette='bright')
plt.title('UMAP Projection of HLA Peptide Data')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig('/home/Hamda.Alhosani/umap_projection_locus.png') 
plt.show()


category = data['category']
plt.figure(figsize=(12, 10))
sns.scatterplot(x=umap_results[:, 0], y=umap_results[:, 1], hue=category, palette='bright')
plt.title('UMAP Projection of HLA Peptide Data')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig('/home/Hamda.Alhosani/umap_projection_category.png') 
plt.show()


