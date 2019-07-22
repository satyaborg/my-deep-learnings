
# Tabular Data

Neural nets for Tabular Data! Logistic Regression, RF, XGB are traditionally used. However NN can be used for analysing Tabular Data.

* Lot of the hand created features were not necessary vs ML algorithms
* Pinterest replaced their algo with NN



```
from fastai import *
from fastai.tabular import *
```


```
path = untar_data(URLs.ADULT_SAMPLE)
```


```
df = pd.read_csv(path/'adult.csv')
```


```
df.head(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>49</td>
      <td>Private</td>
      <td>101320</td>
      <td>Assoc-acdm</td>
      <td>12.0</td>
      <td>Married-civ-spouse</td>
      <td>NaN</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>1902</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;=50k</td>
    </tr>
  </tbody>
</table>
</div>




```
list(df.columns)
```




    ['age',
     'workclass',
     'fnlwgt',
     'education',
     'education-num',
     'marital-status',
     'occupation',
     'relationship',
     'race',
     'sex',
     'capital-gain',
     'capital-loss',
     'hours-per-week',
     'native-country',
     'salary']




```
df.dtypes
```




    age                 int64
    workclass          object
    fnlwgt              int64
    education          object
    education-num     float64
    marital-status     object
    occupation         object
    relationship       object
    race               object
    sex                object
    capital-gain        int64
    capital-loss        int64
    hours-per-week      int64
    native-country     object
    salary             object
    dtype: object



## Independent Variables

As we know variables can be of two types i.e. continuous and categorical.

For continuous, we tend to use embeddings. Think of continuous variables as sending pixels into a neural nets.


```
# dependent variable
y = 'salary'

# Categorical independent variables
cat_x = [ 'workclass',
         'education',
         'marital-status',
         'occupation',
         'relationship',
         'race',
         'sex'
        ]

# Continuous independent variables
cont_x = ['age', 'fnlwgt', 'education-num']

# Processors for preprocessing
procs = [FillMissing, Categorify, Normalize]
```

* Processors are similar to transforms. However, transforms are more of data augmentation.
* FillMissing fills the missing values with medians


Now using datablock api,


```
test = TabularList.from_df(df.iloc[800:1000].copy(),
                          path=path, cat_names=cat_x,
                          cont_names=cont_x)
```


```
data = (TabularList.from_df(df, path=path, cat_names=cat_x,
                           cont_names=cont_x, procs=procs)
                    .split_by_idx(list(range(800,1000)))
                    .label_from_df(cols=y)
                    .add_test(test)
                    .databunch()
       )
```


```
data.show_batch()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>workclass</th>
      <th>education</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>education-num_na</th>
      <th>age</th>
      <th>fnlwgt</th>
      <th>education-num</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Private</td>
      <td>Some-college</td>
      <td>Never-married</td>
      <td>Sales</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Female</td>
      <td>False</td>
      <td>-1.5090</td>
      <td>-0.3589</td>
      <td>-0.0312</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>False</td>
      <td>1.4962</td>
      <td>0.8525</td>
      <td>-0.4224</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <td>Self-emp-not-inc</td>
      <td>5th-6th</td>
      <td>Never-married</td>
      <td>Sales</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Female</td>
      <td>False</td>
      <td>1.7894</td>
      <td>0.2654</td>
      <td>-2.7692</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <td>Private</td>
      <td>Prof-school</td>
      <td>Never-married</td>
      <td>Tech-support</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>False</td>
      <td>-0.5561</td>
      <td>-0.0637</td>
      <td>1.9245</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Never-married</td>
      <td>Handlers-cleaners</td>
      <td>Own-child</td>
      <td>White</td>
      <td>Male</td>
      <td>False</td>
      <td>-0.2629</td>
      <td>-0.0825</td>
      <td>-0.4224</td>
      <td>&lt;50k</td>
    </tr>
  </tbody>
</table>


> Note : Layers is where we define our NN architecture.


```
learn = tabular_learner(data, layers=[200,100], metrics=accuracy)
learn.fit(1, 1e-2)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.365545</td>
      <td>0.373556</td>
      <td>0.830000</td>
      <td>00:07</td>
    </tr>
  </tbody>
</table>


We got an accuracy of 83% (with the default hyperparameters).
