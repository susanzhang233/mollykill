# nail-it-molecule

## Project Proposal

### Abstract

## Project Proposal

### Abstract

This project is aimed to building a model that would aid drug design processes. The project is expected to use machine learning models to determine the inhibition patterns for a specific protein(what biochemical factors contributes more to have the protein inhibited, solubility? binding energy? etc). The model should develope a reasonable way to learn molecules either graphically or numerically. Then, it would check over a larger dataset of molecules to look for some molecules that might work as a novel inhibitor for the protein.

### Planned Deliverables

- Full Success
 
 The model would be available as a package or an accessible website. If of a package form, it would provide the basic functions regarding drug designing process, including functions for data loading, generating outputs, a model created by tensorflow that could be generalized for other proteins. If of the website form, it should be able to prompt the users for inputs, along with some flexible parameters, then, connect to an online compiler for data training, at last providing the output to the user. Useful comments should be included to make the model more user-friendly. Some explanations and research regarding the feature selection and model building process should also be included.


- Partial Success
 
 A somewhat novel methodology is made available to readers. People with access to the code should be able to understand the overall logistics for creating the model. They might not be able to run the code in one hit, but should be able to after some copy-pasting, therefore, they should at least has some understandings of python and machine learning. However, thorough exploratory analysis for feature selection of molecules should still be conducted.


- Oops..
 
 Hope not!

### Resources Required

1. Training database: set of existing(i.e. reported) inhibitors for a specific molecule, could be obtained from CHEMbl, ZINC, kaggle, etc. For example, if we want to look for small molecule inhibitor targeting the covid-19 pathway protein, the ultimate dataset might come from multiple sources: reported covid drugs, drugs already on clinical trials, common antibiotics, effective molecules in chinese traditional medicine, etc. 

Some potential datasets: 
    
   >A databank on kaggle with possible molecules for covid drug: 
https://www.kaggle.com/priteshraj10/coronavirus-covid19-drug-discovery

   >the structure for the only drug that is actually put into use currently: https://pubchem.ncbi.nlm.nih.gov/compound/Remdesivir
   
   >databank for browsing drugs of a specific target:
https://www.ebi.ac.uk/chembl/g/#browse/drugs

2. Database for rewarding: a larger dataset of possible molecules to be screened, could also be obtained as from above.

Some potential datasets: 

>fda approved drugs: https://zinc.docking.org/substances/subsets/endogenous+fda/

The above website contains comprehensive aggregation of all reported molecules, and could be flexibly queried.

However, applying the model to covid-19 is only under thought process stage for me right now. Some viable solutions includes, conducting more research on the covid-19 disease causing pathway, and locate a possible protein target for drug, or choosing another protein to demonstrate the methodology. Indeed, to maximize the likelihood that this model *makes sense* biochemically, more research is needed.

### Tools/Skills Required

 - database management(I believe the lectures about sql is enough, as I render precise data selection before putting into the model for construction to be also important)
 - ChemicalChecker, a package that creates a chemical space that vectorize molecules, also able to predict the vector representation of an unpresent molecule in the space, might be used for data selection or else. https://chemicalchecker.org/
 - RDkit, a useful package for extracting biochemical features of a molecule. https://rdkit.org/docs/api-docs.html
 - dgl life-sci, graph neural network package that could be applied to lifescience elements https://lifesci.dgl.ai/index.html
 - tensorflow, would help in creating graphic neural network models(hope to learn more about them in lectures)

### Risks

### Ethics





### Tentative Timeline

- [ ] 1. Conduct pre-research on existing model. Sum their advantages and possible points for improvements
- [ ] 2. Obtain a dataset of molecule inhibitors of decent amount(approx 5-100) for a specific molecule. Find a way to represent these molecules for following processes
- [ ] 3. Find a way to numerically represent the molecules, and conduct feature selection with the representations.= Would possibly include usage of the package mentioned here https://chemicalchecker.org/
- [ ] 4. Try multiple models, and possibly using multiple models with tensorflow(currently a bit confused on this portion as I'm not so sure about what tensorflow could achieve that scikitlearn could not), to create an ultimate model that effectively uses all the molecules for training
- [ ] 5. Create the rewarding function based on previous training process
- [ ] 6. User-friendlirize the model(maybe generating website access?)
- [ ] 7. Modify each steps to enable the model to be appliable for other proteins generally 

>Timeline

| week1 | week2 | week3 | week4 | week5 | week6 |
| ---- | ---- | ---- | ---- | ---- | ---- |
| 1 & 2 & 3 | 3 & 4 | 4 | 4 & 5 | 6 & 7 | 7 |








| Syntax      | Description | Test Text     |
| :---        |    :----:   |          ---: |
| Header      | Title       | Here's this   |
| Paragraph   | Text        | And more      |