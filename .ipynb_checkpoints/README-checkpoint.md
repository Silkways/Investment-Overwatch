IN PROGRESS

# Investment-Overwatch


## Project Premise 

The goal of this project is to screen global publicly traded companies using unsupervised machine learning.

    - Number of companies: approximately 8,500 
    - Markets include: USA, Australia, Germany, France, Canada, United Kingdom, Norway, Sweden, Denmark, Singapore, Netherlands, Hong Kong, Spain, Italy, Belgium
    - Sectors include: Autos, Mining, Biotech, Healthcare, Consumer goods, Financial Service, Energy etc.

Clustering algorithms are primarily used in this project for segmentation of global stocks into potential study groups to complement fundamental company anlaysis. Our aim is to identify investment opportunities by observing clusters of companies with unique financial features and systematize the company screening process for an investment company.


## Project Structure

Three main notebooks for collection, preprocessing and modelling of data. The main notebook contains a extended altered version of all three notebooks. 

Folders: 

    - visualizations: contains some of graphs and images used to display the work
    - api_data: contains data from API 
    - metric_description: description of the financial features from Finnhub's API documentation
    - archived: files deleted in the course of the project
    
Files: 

    - data_collection.ipynb: Contains examples of API loops to collect financial metrics of companies + additional finnhub features (company news, CEO compensation etc.) 
    - data_preprocessing.ipynb: Conatains data cleaning, imputation and initial observations -> output of this is 'clean_data.csv'
    - analysis_modelling.ipynb: Contains PCA and Clustering algorithms to group companies according to multiple financial dimensions

## Data Sources

The primary source of data is Finnhub, a free platform where you can access a wide range of financial data through their API. Data structure is accessible at a ticker/symbol level. One symbol corresponds to one company. To access data for mulitple companies we require to loop through a list of symbols and stay within the API call limits. We use a list of tickers from Yahoo Finance saved in the ticker_data folder. 

For more insighta on Finnhub, here is an introductory blog: https://medium.com/@augustin.goudet/introduction-to-finnhub-97c2117dd9a9

## Data Description


## Exploratory Data Analysis


## Modelling - PCA / KMeans Clustering / Agglomerative Clustering


## Interpretation


## Limitations Encountered


## Future Work

Data iteration
Sequential Clustering 

## References 





