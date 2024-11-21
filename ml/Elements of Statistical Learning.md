# Chapter 2
i only started taking notes on Obsidian at this point :P

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
- We can use least squares to estimate the parameters $\theta$ in $f_{\theta}$ even if $$f_{\theta}(x) = \sum_{k=1} ^K h_k(x) \theta_k$$ this is called a **linear basis expansion**
- Least squares is convenient but is not the only criterion and does not make sense everywhere
	- __Maximum likelihood estimation__: suppose we have random samples $y_i$, $i = 1, \ldots, N$ from a density $\text{Pr}_{\theta}(y)$
	- Consider the log-probability of the observed sample $$L(\theta) \sum_{i=1} ^N \log \text{Pr}_{\theta}(y_i)$$
	- Principle of maximum likelihood: assumes that the most reasonable values for $\theta$ are those for which the probability of the observed sample is largest
		- Least squares for additive error model $Y = f_{\theta}(X) + \epsilon$ with $\epsilon \sim N(0, \sigma^2)$ is equivalent to maximum likelihood using conditional likelihood, e.g. $$Pr(Y|X, \theta) = N(f_{\theta}(X), \sigma^2)$$
	- More interesting: multinomial likelihood for $\text{Pr}(G|X)$ for qualitative output $G$
		- Consider model $\text{Pr}(G = \mathcal{G}_k | X=x) = p_{k, \theta}(x), k = 1, \ldots, K$ for condition prob. of each class given $X$ (indexed by $\theta$)
		- Log-likelihood of this is $$L(\theta) = \sum_{i=1} ^N \log p_{g_i, \theta} (x_i)$$ when maximized it delivers values of $\theta$ that best conform with this data
  
## 2.7: Structured Regression Models
- Nearest-neighbor and other local methods focus on direct estimation at a point  but face problems in high dimensions
### 2.7.1: Difficulty of the Problem
- Consider $$\text{RSS}(f) = \sum_{i=1} ^N (y_i - f(x_i))^2$$minimizing this leads to infinitely many solutions, since any $\hat{f}$ passing through all $(x_i, y_i)$ is a solution
	- How can we restrict these solutions to a smaller set of functions? Main topic of this book
- Constraints imposed by most learning methods are called __complexity__ restrictions
	- For all inputs $x$ sufficiently close to each other in some metric, $\hat{f}$ exhibits some special structure that the estimator fits to the neighborhood
		- Strength of this constraint is proportional to neighborhood size
- Any method trying to produce locally varying functions in small isotropic neighborhoods will run into problems in high-dimensions (curse of dimensionality)
	- Converse also holds: all methods overcoming curse of dimensionality will have a metric for measuring neighborhoods; however, this will prevent the neighborhood from being simultaneously slow in all directions

## 2.8 Classes of Restricted Estimators
- Various classes of nonparametric regression techniques
	- Some techniques can be in multiple classes

### 2.8.1: Roughness Penalty and Bayesian Methods
- Class of functions controlled by a **roughness penalty** $$\text{PRSS}(f; \lambda) = \text{RSS}(f) + \lambda J(f)$$
  where $J(f)$ is the **roughness penalty** (large for functions that vary over small regions over input space)
  - **Example**: cubic smoothing spline for one-dimensional inputs $$\text{PRSS}(f; \lambda) = \sum_{i=1} ^N (y_i - f(x_i))^2 + \lambda \int[f''(x)]^2 \ dx$$
	  - Roughness penalty controls large values of $f''(x)$ and dictated by $\lambda \geq 0$
	  - $\lambda = \infty$ only permits functions linear in $x$
  - Also known as **regularization methods**; express prior belief of data exhibiting certain smooth behavior

### 2.8.2: Kernel Methods and Local Regression
- Explicitly providing estimates of regression function or conditional expectation by specifying nature of local neighborhood via a **kernel function** $K_{\lambda]}(x_0, x)$
- General framework: $$\text{RSS}(f_{\theta}, x_0) = \sum_{i=1} ^N K_{\lambda} (x_0, x_i)(y_i - f_{\theta}(x_i))^2$$
  where $f_{\theta}$ is a parametrized function
  - Nearest-neighbor methods are a form of a kernel method with a more data-dependent metric

### 2.8.3: Basis Functions and Dictionary Methods
- $f$ is a linear expansion of basis functions: $$f_{\theta}(x) = \sum_{m=1} ^M \theta_m h_m(x)$$
- 
