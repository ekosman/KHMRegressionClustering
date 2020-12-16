# KHMRegressionClustering 

## Installation
* Clone this repository
* `cd .`
* Install via `pip install -e .`
## Usage example

```python
from khm_rc.KHM import KHM

# Prepare some data to fit the model on
coeff1 = [0.2, -0.5, 5]
coeff2 = [-0.2, -10, -5]
coeff3 = [0, 4, 0]
basis1 = [lambda x: x[0]**2, lambda x: x[0], lambda x: 1]
basis2 = [lambda x: x[0]**2, lambda x: x[0], lambda x: 1]
basis3 = [lambda x: x[0]**2, lambda x: x[0], lambda x: 1]

x1 = np.linspace(-50, 50, 100).reshape(-1, 1)
x2 = np.linspace(-50, 50, 100).reshape(-1, 1)
x3 = np.linspace(-50, 50, 100).reshape(-1, 1)
y1 = np.array([calc_f(basis=basis1, coeff=coeff1, x=x_i) + np.random.randn()*20 for x_i in x1])
y2 = np.array([calc_f(basis=basis2, coeff=coeff2, x=x_i) + np.random.randn()*20 for x_i in x2])
y3 = np.array([calc_f(basis=basis3, coeff=coeff3, x=x_i) + np.random.randn()*20 for x_i in x3])

x = np.concatenate([x1, x2, x3])
y = np.concatenate([y1, y2, y3])

# Fit the model. trials is used in order to fit the model multiple times and choose the iteration with best performance
# This is because the model can converge to different local minimas for different initializations
model = KHM(function_basis=[basis1, basis2, basis3])
model.fit(x=x, y=y, max_iterations=10, trials=10, verbose='iteration')
print(repr(model))
```

After fitting the model, visualize it:
```python
# Prepare The x coordinates for each functions as a linear spaced dots
# Later, evaluate each function
x_1 = np.linspace(-80, 80, 100).reshape(-1, 1)
x_2 = np.linspace(-80, 80, 100).reshape(-1, 1)
x_3 = np.linspace(-80, 80, 100).reshape(-1, 1)
res1 = model.calc_kth_function(k=0, x=x_1)
res2 = model.calc_kth_function(k=1, x=x_2)
res3 = model.calc_kth_function(k=2, x=x_3)

# Plot
plt.figure()
plt.scatter(x1, y1, label='f1', s=15)
plt.scatter(x2, y2, label='f2', s=15)
plt.scatter(x3, y3, label='f3', s=15)
plt.plot(x_1, res1)
plt.plot(x_2, res2)
plt.plot(x_3, res3)
plt.grid()
plt.show()
```
<img src=figures/results.png width="600"/>

## Cite this repository

```
@misc{ekosman_khmrc_2020,
    author       = {Eitan Kosman},
    title        = {{K-Harmonic Means Regression Clustering in Python}},
    month        = September,
    year         = 2020,
    url          = {https://github.com/ekosman/KHMRegressionClustering}
    }
```
