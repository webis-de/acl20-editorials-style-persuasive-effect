# Paper: Analyzing the Persuasive Effect of Style in News Editorial Argumentation

This is the code for the paper *Analyzing the Persuasive Effect of Style in News Editorial Argumentation*.

Roxanne El Baff, Henning Wachstmuth, Khalid Al-Khatib and Benno Stein


      @InProceedings{elbaff:2020,
        author =              {Roxanne {El Baff} and Henning Wachsmuth and Khalid Al-Khatib and Benno Stein},
        booktitle =           {The 58th annual meeting of the Association for Computational Linguistics (ACL) },
        month =               jul,
        publisher =           {ACL},
        site =                {Seattle, USA},
        title =               {{Analyzing the Persuasive Effect of Style in News Editorial Argumentation}},
        year =                2020
      }
    
[[Paper Link](https://www.aclweb.org/anthology/2020.acl-main.287/)] 


### Dependency
Python version: 3.6.

The readme file gives an overview on which part of the code is responsible for which section in the paper. We share our trained models, results of significance tests, and the values of the features extracted as detailed below.


### Overall Experiments

  - The two main notebook is **section5-prediction of-persuasive-effects.ipynb** and **section6-identification-of-style-patterns.ipynb** are used for running the experiment for section 5 and 6 respectively. 
  - loader.py: loads the NYT corpus, calculate majority effect for each ideoloy and other features that are deduced from Webis Editorial-Quality-18 (WEQ) corpus (El Baff et al.,
  2018). 
  - classification using pre-trained BERT is in **dismissed_experiments.ipynb**.
  - utility.py: contains the code that helped clean and detect duplicate annotated articles in WEQ.

  **NOTE**: unzip the json files under the *data* folder in order to reproduce the experiments


### Section 3 - Style Features and Section 4 - Data

#### *code*
   - text_miner.py: preprocesses data (e.g. tokenization, lemma, etc. ).
   - features.py: extracts n-gram lemma features
   - lexicon.py: extracts lexicon based features ( NRC, arguing, ...)
   
#### *data*
  - annotations folder: Contains the results of the OpenFinder classification for *arguing* lexicon and Subjective/Objective classifications: OpinionFinder 2.0 (Riloff and
Wiebe, 2003; Wiebe and Riloff, 2005)
  - corpus folder: contains the NYT corpus (it should be added manually because of the copy rights for NYT - check readme file under this folder).  
  - data folder: contains the features extracted from the corpus (e.g. lemma, Webis ADUs, LIWC features)
  - lexicon folder: contains all the lexicons used in this paper:
          -  NRC Mohammad and Turney (2013)
          -  MPQA Arguing Somasundaran et al. (2007)
          
          
### Section 5 - Prediction of Persuasive Effects

#### *code*
 Experiments are ran in notebook **section5-prediction of-persuasive-effects.ipynb**:  
  - deals with training/testing the WEQ corpus, especifically *run_experiments*. 
  - runs significance tests for "baseline" vs. "content", "baseline" vs. "best of style", "content" vs. "best of style". The main function is *run_experiments_with_test_repetition*.
  - This notebook mainly uses machine_learning.py which does the following:
              - removes outliers
              - normalizes data
              - trains and evaluates model
              - saves trained models
              
#### *data*
  - models folder: 
      - contains the trained models (style only, style+content and content only), for both ideologies, saved as pickle objects
      - contains the macro/micro F1 scores for all the feature set combinations along with the hyperparemeters tuned on the training set.


### Section 6 - Identification of Style Patterns

main notebook is **section6-identification-of-style-patterns.ipynb** along with the following python files of eacht substep.

#### 1.  Extract the style features from Section 3.

##### *code*
 - lexicon.py: extracts lexicon based features ( NRC, arguing, ...)


#### 2. Perform a cluster analysis on the style features

##### *code*
 - cluster_analysis.py: used for using k-mean /cosine k-means, getting optimal k.
 - significance_testing.py: conducts significance tests. It is used to check significany for each feature between the clusters to determine *descriminative features*.

##### *data*
- clustering folder: contains the results of the clustering analysis results (barchart, elbow graph).
 - significance_tests folders: significance tests results for each feature used for clustering. Based on it, the discriminative features were selected.


#### 3 . Identification of Style Patterns

##### *code*

 - using the notebook **section6-identification-of-style-patterns.ipynb**. 

##### *data*
  - output folder: flows.csv
