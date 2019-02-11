
# Welcome to Jupyter!


```python
import numpy as np
import pandas as pd
a = pd.Series([1,2,3,np.nan,5],index=['a','b','c','d','e'])
a
```




    a    1.0
    b    2.0
    c    3.0
    d    NaN
    e    5.0
    dtype: float64




```python
# create a data frame using Datetime as Index
dates = pd.date_range('20190211',periods=9)
df = pd.DataFrame(np.random.randn(9,4),index=dates, columns = list('cvbn'))
df
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
      <th>c</th>
      <th>v</th>
      <th>b</th>
      <th>n</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-02-11</th>
      <td>0.428653</td>
      <td>-0.859452</td>
      <td>-0.964801</td>
      <td>0.070963</td>
    </tr>
    <tr>
      <th>2019-02-12</th>
      <td>-0.255813</td>
      <td>-0.591874</td>
      <td>0.135555</td>
      <td>0.201801</td>
    </tr>
    <tr>
      <th>2019-02-13</th>
      <td>-0.752381</td>
      <td>-0.347708</td>
      <td>-1.823415</td>
      <td>0.109480</td>
    </tr>
    <tr>
      <th>2019-02-14</th>
      <td>0.232037</td>
      <td>-0.235704</td>
      <td>0.037106</td>
      <td>0.215268</td>
    </tr>
    <tr>
      <th>2019-02-15</th>
      <td>0.402103</td>
      <td>-0.010526</td>
      <td>-0.452265</td>
      <td>-0.902612</td>
    </tr>
    <tr>
      <th>2019-02-16</th>
      <td>-0.101289</td>
      <td>-0.506196</td>
      <td>0.713672</td>
      <td>-1.597138</td>
    </tr>
    <tr>
      <th>2019-02-17</th>
      <td>-2.353637</td>
      <td>1.180212</td>
      <td>-0.881545</td>
      <td>-0.105711</td>
    </tr>
    <tr>
      <th>2019-02-18</th>
      <td>1.520008</td>
      <td>-0.354201</td>
      <td>1.569270</td>
      <td>-0.134891</td>
    </tr>
    <tr>
      <th>2019-02-19</th>
      <td>0.572177</td>
      <td>-0.510602</td>
      <td>0.036388</td>
      <td>0.145626</td>
    </tr>
  </tbody>
</table>
</div>




```python
np.random.randn(3,2)
```




    array([[ 1.42931572,  0.81621294],
           [-0.6301987 , -0.23648009],
           [ 0.59593101, -0.29783017]])




```python
pd.Timestamp('20190211')

```




    Timestamp('2019-02-11 00:00:00')




```python
df2 = pd.DataFrame({'A':1.,
                   'B':pd.Timestamp('20190211'),
                   'C':pd.Categorical(['test','train','test','train'])})
df2
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2019-02-11</td>
      <td>test</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>2019-02-11</td>
      <td>train</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>2019-02-11</td>
      <td>test</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>2019-02-11</td>
      <td>train</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.dtypes
```




    A           float64
    B    datetime64[ns]
    C          category
    dtype: object




```python
df.dtypes
```




    c    float64
    v    float64
    b    float64
    n    float64
    dtype: object




```python
df2.A
```




    0    1.0
    1    1.0
    2    1.0
    3    1.0
    Name: A, dtype: float64




```python
df2.abs

```




    <bound method NDFrame.abs of      A          B      C
    0  1.0 2019-02-11   test
    1  1.0 2019-02-11  train
    2  1.0 2019-02-11   test
    3  1.0 2019-02-11  train>




```python
df2.add
```




    <bound method _arith_method_FRAME.<locals>.f of      A          B      C
    0  1.0 2019-02-11   test
    1  1.0 2019-02-11  train
    2  1.0 2019-02-11   test
    3  1.0 2019-02-11  train>




```python
df2.add_prefix('hey')
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
      <th>heyA</th>
      <th>heyB</th>
      <th>heyC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2019-02-11</td>
      <td>test</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>2019-02-11</td>
      <td>train</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>2019-02-11</td>
      <td>test</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>2019-02-11</td>
      <td>train</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.add_suffix('SchoolOfAiDean')
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
      <th>ASchoolOfAiDean</th>
      <th>BSchoolOfAiDean</th>
      <th>CSchoolOfAiDean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2019-02-11</td>
      <td>test</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>2019-02-11</td>
      <td>train</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>2019-02-11</td>
      <td>test</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>2019-02-11</td>
      <td>train</td>
    </tr>
  </tbody>
</table>
</div>




```python
l = df2.align
m = df2.all
n = df2.all
o = df2.any
p = df2.append
q = df2.apply
r = df2.applymap
s = df2.C
print(l)
print(m)
print(n)
print(o)
print(p)
print(q)
print(r)
print(s)
```

    <bound method DataFrame.align of      A          B      C
    0  1.0 2019-02-11   test
    1  1.0 2019-02-11  train
    2  1.0 2019-02-11   test
    3  1.0 2019-02-11  train>
    <bound method DataFrame.all of      A          B      C
    0  1.0 2019-02-11   test
    1  1.0 2019-02-11  train
    2  1.0 2019-02-11   test
    3  1.0 2019-02-11  train>
    <bound method DataFrame.all of      A          B      C
    0  1.0 2019-02-11   test
    1  1.0 2019-02-11  train
    2  1.0 2019-02-11   test
    3  1.0 2019-02-11  train>
    <bound method DataFrame.any of      A          B      C
    0  1.0 2019-02-11   test
    1  1.0 2019-02-11  train
    2  1.0 2019-02-11   test
    3  1.0 2019-02-11  train>
    <bound method DataFrame.append of      A          B      C
    0  1.0 2019-02-11   test
    1  1.0 2019-02-11  train
    2  1.0 2019-02-11   test
    3  1.0 2019-02-11  train>
    <bound method DataFrame.apply of      A          B      C
    0  1.0 2019-02-11   test
    1  1.0 2019-02-11  train
    2  1.0 2019-02-11   test
    3  1.0 2019-02-11  train>
    <bound method DataFrame.applymap of      A          B      C
    0  1.0 2019-02-11   test
    1  1.0 2019-02-11  train
    2  1.0 2019-02-11   test
    3  1.0 2019-02-11  train>
    0     test
    1    train
    2     test
    3    train
    Name: C, dtype: category
    Categories (2, object): [test, train]



```python
df2.xs
```




    <bound method NDFrame.xs of      A          B      C
    0  1.0 2019-02-11   test
    1  1.0 2019-02-11  train
    2  1.0 2019-02-11   test
    3  1.0 2019-02-11  train>




```python
df2.bool

```




    <bound method NDFrame.bool of      A          B      C
    0  1.0 2019-02-11   test
    1  1.0 2019-02-11  train
    2  1.0 2019-02-11   test
    3  1.0 2019-02-11  train>




```python
df2.boxplot
```




    <bound method boxplot_frame of      A          B      C
    0  1.0 2019-02-11   test
    1  1.0 2019-02-11  train
    2  1.0 2019-02-11   test
    3  1.0 2019-02-11  train>




```python
df2.clip
```




    <bound method NDFrame.clip of      A          B      C
    0  1.0 2019-02-11   test
    1  1.0 2019-02-11  train
    2  1.0 2019-02-11   test
    3  1.0 2019-02-11  train>




```python
df2.clip_lower
```




    <bound method NDFrame.clip_lower of      A          B      C
    0  1.0 2019-02-11   test
    1  1.0 2019-02-11  train
    2  1.0 2019-02-11   test
    3  1.0 2019-02-11  train>




```python
df2.clip_upper

```




    <bound method NDFrame.clip_upper of      A          B      C
    0  1.0 2019-02-11   test
    1  1.0 2019-02-11  train
    2  1.0 2019-02-11   test
    3  1.0 2019-02-11  train>




```python
df2.columns
```




    Index(['A', 'B', 'C'], dtype='object')




```python
df2.combine
```




    <bound method DataFrame.combine of      A          B      C
    0  1.0 2019-02-11   test
    1  1.0 2019-02-11  train
    2  1.0 2019-02-11   test
    3  1.0 2019-02-11  train>




```python
df2.compound
```




    <bound method NDFrame._add_numeric_operations.<locals>.compound of      A          B      C
    0  1.0 2019-02-11   test
    1  1.0 2019-02-11  train
    2  1.0 2019-02-11   test
    3  1.0 2019-02-11  train>




```python
df2.consolidate
```




    <bound method NDFrame.consolidate of      A          B      C
    0  1.0 2019-02-11   test
    1  1.0 2019-02-11  train
    2  1.0 2019-02-11   test
    3  1.0 2019-02-11  train>




```python
df.head()
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
      <th>c</th>
      <th>v</th>
      <th>b</th>
      <th>n</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-02-11</th>
      <td>0.428653</td>
      <td>-0.859452</td>
      <td>-0.964801</td>
      <td>0.070963</td>
    </tr>
    <tr>
      <th>2019-02-12</th>
      <td>-0.255813</td>
      <td>-0.591874</td>
      <td>0.135555</td>
      <td>0.201801</td>
    </tr>
    <tr>
      <th>2019-02-13</th>
      <td>-0.752381</td>
      <td>-0.347708</td>
      <td>-1.823415</td>
      <td>0.109480</td>
    </tr>
    <tr>
      <th>2019-02-14</th>
      <td>0.232037</td>
      <td>-0.235704</td>
      <td>0.037106</td>
      <td>0.215268</td>
    </tr>
    <tr>
      <th>2019-02-15</th>
      <td>0.402103</td>
      <td>-0.010526</td>
      <td>-0.452265</td>
      <td>-0.902612</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail()
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
      <th>c</th>
      <th>v</th>
      <th>b</th>
      <th>n</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-02-15</th>
      <td>0.402103</td>
      <td>-0.010526</td>
      <td>-0.452265</td>
      <td>-0.902612</td>
    </tr>
    <tr>
      <th>2019-02-16</th>
      <td>-0.101289</td>
      <td>-0.506196</td>
      <td>0.713672</td>
      <td>-1.597138</td>
    </tr>
    <tr>
      <th>2019-02-17</th>
      <td>-2.353637</td>
      <td>1.180212</td>
      <td>-0.881545</td>
      <td>-0.105711</td>
    </tr>
    <tr>
      <th>2019-02-18</th>
      <td>1.520008</td>
      <td>-0.354201</td>
      <td>1.569270</td>
      <td>-0.134891</td>
    </tr>
    <tr>
      <th>2019-02-19</th>
      <td>0.572177</td>
      <td>-0.510602</td>
      <td>0.036388</td>
      <td>0.145626</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.index
```




    DatetimeIndex(['2019-02-11', '2019-02-12', '2019-02-13', '2019-02-14',
                   '2019-02-15', '2019-02-16', '2019-02-17', '2019-02-18',
                   '2019-02-19'],
                  dtype='datetime64[ns]', freq='D')




```python
df.describe()
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
      <th>c</th>
      <th>v</th>
      <th>b</th>
      <th>n</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>9.000000</td>
      <td>9.000000</td>
      <td>9.000000</td>
      <td>9.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.034238</td>
      <td>-0.248450</td>
      <td>-0.181115</td>
      <td>-0.221913</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.072842</td>
      <td>0.585801</td>
      <td>0.995938</td>
      <td>0.620183</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.353637</td>
      <td>-0.859452</td>
      <td>-1.823415</td>
      <td>-1.597138</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.255813</td>
      <td>-0.510602</td>
      <td>-0.881545</td>
      <td>-0.134891</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.232037</td>
      <td>-0.354201</td>
      <td>0.036388</td>
      <td>0.070963</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.428653</td>
      <td>-0.235704</td>
      <td>0.135555</td>
      <td>0.145626</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.520008</td>
      <td>1.180212</td>
      <td>1.569270</td>
      <td>0.215268</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.T
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
      <th>2019-02-11 00:00:00</th>
      <th>2019-02-12 00:00:00</th>
      <th>2019-02-13 00:00:00</th>
      <th>2019-02-14 00:00:00</th>
      <th>2019-02-15 00:00:00</th>
      <th>2019-02-16 00:00:00</th>
      <th>2019-02-17 00:00:00</th>
      <th>2019-02-18 00:00:00</th>
      <th>2019-02-19 00:00:00</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>c</th>
      <td>0.428653</td>
      <td>-0.255813</td>
      <td>-0.752381</td>
      <td>0.232037</td>
      <td>0.402103</td>
      <td>-0.101289</td>
      <td>-2.353637</td>
      <td>1.520008</td>
      <td>0.572177</td>
    </tr>
    <tr>
      <th>v</th>
      <td>-0.859452</td>
      <td>-0.591874</td>
      <td>-0.347708</td>
      <td>-0.235704</td>
      <td>-0.010526</td>
      <td>-0.506196</td>
      <td>1.180212</td>
      <td>-0.354201</td>
      <td>-0.510602</td>
    </tr>
    <tr>
      <th>b</th>
      <td>-0.964801</td>
      <td>0.135555</td>
      <td>-1.823415</td>
      <td>0.037106</td>
      <td>-0.452265</td>
      <td>0.713672</td>
      <td>-0.881545</td>
      <td>1.569270</td>
      <td>0.036388</td>
    </tr>
    <tr>
      <th>n</th>
      <td>0.070963</td>
      <td>0.201801</td>
      <td>0.109480</td>
      <td>0.215268</td>
      <td>-0.902612</td>
      <td>-1.597138</td>
      <td>-0.105711</td>
      <td>-0.134891</td>
      <td>0.145626</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.sort_index(axis=1, ascending=False)
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
      <th>v</th>
      <th>n</th>
      <th>c</th>
      <th>b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-02-11</th>
      <td>-0.859452</td>
      <td>0.070963</td>
      <td>0.428653</td>
      <td>-0.964801</td>
    </tr>
    <tr>
      <th>2019-02-12</th>
      <td>-0.591874</td>
      <td>0.201801</td>
      <td>-0.255813</td>
      <td>0.135555</td>
    </tr>
    <tr>
      <th>2019-02-13</th>
      <td>-0.347708</td>
      <td>0.109480</td>
      <td>-0.752381</td>
      <td>-1.823415</td>
    </tr>
    <tr>
      <th>2019-02-14</th>
      <td>-0.235704</td>
      <td>0.215268</td>
      <td>0.232037</td>
      <td>0.037106</td>
    </tr>
    <tr>
      <th>2019-02-15</th>
      <td>-0.010526</td>
      <td>-0.902612</td>
      <td>0.402103</td>
      <td>-0.452265</td>
    </tr>
    <tr>
      <th>2019-02-16</th>
      <td>-0.506196</td>
      <td>-1.597138</td>
      <td>-0.101289</td>
      <td>0.713672</td>
    </tr>
    <tr>
      <th>2019-02-17</th>
      <td>1.180212</td>
      <td>-0.105711</td>
      <td>-2.353637</td>
      <td>-0.881545</td>
    </tr>
    <tr>
      <th>2019-02-18</th>
      <td>-0.354201</td>
      <td>-0.134891</td>
      <td>1.520008</td>
      <td>1.569270</td>
    </tr>
    <tr>
      <th>2019-02-19</th>
      <td>-0.510602</td>
      <td>0.145626</td>
      <td>0.572177</td>
      <td>0.036388</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.sort_values(by='B')
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2019-02-11</td>
      <td>test</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>2019-02-11</td>
      <td>train</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>2019-02-11</td>
      <td>test</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>2019-02-11</td>
      <td>train</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2['A']
```




    0    1.0
    1    1.0
    2    1.0
    3    1.0
    Name: A, dtype: float64




```python
df2[0:3]
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2019-02-11</td>
      <td>test</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>2019-02-11</td>
      <td>train</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>2019-02-11</td>
      <td>test</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['20190213':'20190218']
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
      <th>c</th>
      <th>v</th>
      <th>b</th>
      <th>n</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-02-13</th>
      <td>-0.752381</td>
      <td>-0.347708</td>
      <td>-1.823415</td>
      <td>0.109480</td>
    </tr>
    <tr>
      <th>2019-02-14</th>
      <td>0.232037</td>
      <td>-0.235704</td>
      <td>0.037106</td>
      <td>0.215268</td>
    </tr>
    <tr>
      <th>2019-02-15</th>
      <td>0.402103</td>
      <td>-0.010526</td>
      <td>-0.452265</td>
      <td>-0.902612</td>
    </tr>
    <tr>
      <th>2019-02-16</th>
      <td>-0.101289</td>
      <td>-0.506196</td>
      <td>0.713672</td>
      <td>-1.597138</td>
    </tr>
    <tr>
      <th>2019-02-17</th>
      <td>-2.353637</td>
      <td>1.180212</td>
      <td>-0.881545</td>
      <td>-0.105711</td>
    </tr>
    <tr>
      <th>2019-02-18</th>
      <td>1.520008</td>
      <td>-0.354201</td>
      <td>1.569270</td>
      <td>-0.134891</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc[dates[0]]
```




    c    0.428653
    v   -0.859452
    b   -0.964801
    n    0.070963
    Name: 2019-02-11 00:00:00, dtype: float64




```python
df2.loc[:,['A','B']]
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
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2019-02-11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>2019-02-11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>2019-02-11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>2019-02-11</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.loc['20190214':'20190219',['A','B']]
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
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
df.at[dates[0],'c']
```




    0.4286528969227465




```python

```
