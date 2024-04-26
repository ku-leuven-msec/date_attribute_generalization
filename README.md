# Advanced methods for generalizing time and duration during dataset anonymization
This repository contains the results and the implementation of the hierarchy creation methods and evaluation used in the paper "Advanced methods for generalizing time and duration during dataset anonymization".

## Abstract
Time is an often recurring quasi-identifying attribute in many datasets. Anonymizing such datasets requires generalizing the time attribute(s) in the dataset.  Examples are start dates and durations, which are traditionally generalized leading to intervals that do not embrace the relation between time attributes. This paper presents advanced methods for creating generalization hierarchies for time data. We propose clustering-based and Mondrian-based techniques to construct generalization hierarchies. These approaches take into account the relation between different time attributes and are designed to improve the utility of the anonymized data. We implemented these methods and conducted a set of experiments comparing them to traditional generalization strategies. The results show that our proposed methods improve the utility of the data for both statistical analysis and machine learning applications. Our approach demonstrates a significant increase in hierarchy quality and configuration flexibility, demonstrating the potential of our advanced techniques over existing methods.
## Repository Content
### datasets
Contains the used airfare dataset.
### hierarchy_creators
Contains the code used in creating various hierarchies mentioned in the paper.
### evaluation
Contains all the code to evaluate count queries and run the machine learning task.
### traditional_hierarchies
Contains the traditional hierarchies using calendar and range generalizations.
### output
Contains all the results in the following structure:
* queries: for each hierarchy and assumed distribution the query error results in each hierarchy level.
* ml: contains results of the machine learning utility over 5 folds. Each fold contains the results for each hierarchy and all pre-processors. The test_index.csv file contains the indices of the rows used as test_set for the current fold.