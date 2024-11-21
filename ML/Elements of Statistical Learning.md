# Chapter 2

## 2.6: Statistical Models, Supervised Learning, Function Approximation

- **Goal**: find a useful approximation $\hat{f}(x)$ to $f(x)$

### 2.6.1: Statistical Model for Pr(X, Y)

- Suppose our data arose from a statistical model $$Y = f(X) + \epsilon$$ where $\mathbb{E}(\epsilon|X) = \mathbb{E}(\epsilon) = 0$. This is called an **additive error** model
- Some systems will not have $(X, Y)$ deterministically be determined by $Y = f(X)$; while others do, we will pursue problems in the form $Y = f(X) + \epsilon$

### 2.6.2: Supervised Learning
- Suppose $Y = f(x) + \epsilon$ is a reasonable assumption and the errors are additive
- **Supervised learning** attempts to learn $f$ by example through a **teacher**
	- Training set $\mathcal{T} = (x_i, y_i), i = 1, \ldots, N$
	- Each $x_i$ is an observed input value and fed into a **learning algorithm**, which produces outputs $\hat{f}(x_i)$
	- The learning algorithm aims to minimize the differences $y_i - \hat{f}(x_i)$

### 2.6.3: Function Approximation
- Look at $\{x_i, y_i\}$ as points in a $(p+1)$-dimensional Euclidean space
- $f(x)$ has domain equal to $p$-dimensional input subspace and is in the form $$y_i = f(x_i) + \epsilon_i$$
- For now, assume domain is $\mathbb{R}^p$
- **Goal**: obtain useful approximation to $f(x)$ for all $x$ in some region of $\mathbb{R}^p$ given the data in $\mathcal{T}$
- Many approximations will have associated parameters $\theta$; for example in $f(x) = x^T \beta$, $\theta = \beta$
	- We can use least squares to estimate parameters $\theta$ in $f_{\theta}$ by minimizing $$\text{RSS}(\theta) = \sum_{i=1} ^N (y_i - f_{\theta}(x_i))^2$$ as a function of $\theta$
- 

## 2.7: Structured Regression Models
- Nearest-neighbor and other local methods focus on direct estimation at a point  but face problems in high dimensions
### 2.7.1: Difficulty of the Problem
- Consider $$\text{RSS}(f) = \sum_{i=1} ^N (y_i - f(x_i))^2$$minimizing this leads to infinitely many solutions, snice any $\hat{f}$ passing through all $(x_i, y_i)$ is a solution
- 
  
  
