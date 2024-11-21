# Chapter 1: Markov Chains

## 1.4 Stationary Distributions
- If we assume **aperiodicity**, an irreducible, finite-state Markov chain converges to a stationary distribution $$p^n (x, y) \to \pi(y)$$
- What happens in a Markov chain when the initial state is random? 
  
  $$
	  \begin{align*}
		  P(X_n = j) &= \sum_i P(X_0 = i, X_n = j) \\
		  &= \sum_i P(X_0 = i)P(X_n = j | X_0 = i)
	  \end{align*}
  $$
  If we let $q(i) = P(X_0) = i)$, the last equation becomes 
  
  $$P(X_n = j) = \sum_i q(i) p^n(i, j)$$
  - In other words, we multiply the transition matrix on the left by the vector $q$ of initial probabilities; if there are $k$ states, $p^n(x, y)$ is a $k \times k$ matrix
	  - $q$ is therefore a $1 \times k$ matrix or a "row vector"
  - 