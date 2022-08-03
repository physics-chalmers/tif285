# Statistical inference

<!-- !split -->
## How do you feel about statistics?
<!-- !bpop -->
```{epigraph}
> “There are three kinds of lies: lies, damned lies, and statistics.”

-- Disraeli (attr.): 
```

<!-- !epop -->

<!-- !bpop -->
```{epigraph}
> “If your result needs a statistician then you should design a better experiment.”

-- Rutherford
```

<!-- !epop -->

<!-- !bpop -->
```{epigraph}
> “La théorie des probabilités n'est que le bon sens réduit au calcul”
>
> (rules of statistical inference are an application of the laws of probability)

-- Laplace
```

<!-- !split -->
### Inference
 * Deductive inference. Cause $\to$ Effect. 
 * Inference to best explanation. Effect $\to$ Cause. 
 * Scientists need a way to:
    * Quantify the strength of inductive inferences;
    * Update that quantification as they acquire new data.



<!-- !split -->
### Some history
Adapted from D.S. Sivia {cite}`Sivia2006`

> Although the frequency definition appears to be more objective, its range of validity is also far more limited. For example, Laplace used (his) probability theory to estimate the mass of Saturn, given orbital data that were available to him from various astronomical observatories. In essence, he computed the posterior pdf for the mass M , given the data and all the relevant background information I (such as a knowledge of the laws of classical mechanics): prob(M|{data},I); this is shown schematically in the figure [Fig. 1.2].



<!-- !split -->
<!-- <img src="fig/BayesianBasics/sivia_fig_1_2.png" width=700> -->
![](./figs/sivia_fig_1_2.png)

<!-- !split -->
> To Laplace, the (shaded) area under the posterior pdf curve between $m_1$ and $m_2$ was a measure of how much he believed that the mass of Saturn lay in the range $m_1 \le M \le m_2$. As such, the position of the maximum of the posterior pdf represents a best estimate of the mass; its width, or spread, about this optimal value gives an indication of the uncertainty in the estimate. Laplace stated that: ‘ . . . it is a bet of 11,000 to 1 that the error of this result is not 1/100th of its value.’ He would have won the bet, as another 150 years’ accumulation of data has changed the estimate by only 0.63%!



<!-- !split -->
> According to the frequency definition, however, we are not permitted to use probability theory to tackle this problem. This is because the mass of Saturn is a constant and not a random variable; therefore, it has no frequency distribution and so probability theory cannot be used.
> 
> If the pdf [of Fig. 1.2] had to be interpreted in terms of the frequency definition, we would have to imagine a large ensemble of universes in which everything remains constant apart from the mass of Saturn.



<!-- !split -->
> As this scenario appears quite far-fetched, we might be inclined to think of [Fig. 1.2] in terms of the distribution of the measurements of the mass in many repetitions of the experiment. Although we are at liberty to think about a problem in any way that facilitates its solution, or our understanding of it, having to seek a frequency interpretation for every data analysis problem seems rather perverse.
> For example, what do we mean by the ‘measurement of the mass’ when the data consist of orbital periods? Besides, why should we have to think about many repetitions of an experiment that never happened? What we really want to do is to make the best inference of the mass given the (few) data that we actually have; this is precisely the Bayes and Laplace view of probability.



<!-- !split -->
> Faced with the realization that the frequency definition of probability theory did not permit most real-life scientific problems to be addressed, a new subject was invented — statistics! To estimate the mass of Saturn, for example, one has to relate the mass to the data through some function called the statistic; since the data are subject to ‘random’ noise, the statistic becomes the random variable to which the rules of probability theory can be applied. But now the question arises: How should we choose the statistic? The frequentist approach does not yield a natural way of doing this and has, therefore, led to the development of several alternative schools of orthodox or conventional statistics. The masters, such as Fisher, Neyman and Pearson, provided a variety of different principles, which has merely resulted in a plethora of tests and procedures without any clear underlying rationale. This lack of unifying principles is, perhaps, at the heart of the shortcomings of the cook-book approach to statistics that students are often taught even today.



<!-- !split -->
## Probability density functions (pdf:s)

 * $p(A|B)$ reads “probability of $A$ given $B$”
 * Simplest examples are discrete, but physicists often interested in continuous case, e.g., parameter estimation.
 * When integrated, continuous pdfs become probabilities $\Rightarrow$ pdfs are NOT dimensionless, even though probabilities are.
 * 68%, 95%, etc. intervals can then be computed by integration 
 * Certainty about a parameter corresponds to $p(x) = \delta(x-x_0)$



<!-- !split -->
<!-- ======= pdfs ======= -->
<!-- !split -->
### Properties of PDFs

There are two properties that all PDFs must satisfy. The first one is
positivity (assuming that the PDF is normalized)

$$
0 \leq p(x).
$$
Naturally, it would be nonsensical for any of the values of the domain
to occur with a probability less than $0$. Also,
the PDF must be normalized. That is, all the probabilities must add up
to unity.  The probability of "anything" to happen is always unity. For
discrete and continuous PDFs, respectively, this condition is
\begin{gather*}
\sum_{x_i\in\mathbb D} p(x_i) & =  1,\\
\int_{x\in\mathbb D} p(x)\,dx & =  1.
\end{gather*}



<!-- !split -->
### Important distributions, the uniform distribution
Let us consider some important, univariate distributions.
The first one
is the most basic PDF; namely the uniform distribution
\begin{equation}
p(x) = \frac{1}{b-a}\theta(x-a)\theta(b-x).
\label{eq:unifromPDF}
\end{equation}
For $a=0$ and $b=1$ we have 
\begin{equation*}
p(x) = \left\{
\begin{array}{ll}
1 & x \in [0,1],\\
0 & \mathrm{otherwise}
\end{array}
\right.
\end{equation*}



<!-- !split -->
### Gaussian distribution
The second one is the univariate Gaussian Distribution
\begin{equation*}
p(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp{(-\frac{(x-\mu)^2}{2\sigma^2})},
\end{equation*}
with mean value $\mu$ and standard deviation $\sigma$. If $\mu=0$ and $\sigma=1$, it is normally called the **standard normal distribution**
\begin{equation*}
p(x) = \frac{1}{\sqrt{2\pi}} \exp{(-\frac{x^2}{2})},
\end{equation*}


<!-- !split -->
### Expectation values
Let $h(x)$ be an arbitrary continuous function on the domain of the stochastic
variable $X$ whose PDF is $p(x)$. We define the *expectation value*
of $h$ with respect to $p$ as follows
\begin{equation}
\mathbb{E}_p[h] = \langle h \rangle_p \equiv \int\! h(x)p(x)\,dx
\label{eq:expectation_value_of_h_wrt_p}
\end{equation}
Whenever the PDF is known implicitly, like in this case, we will drop
the index $p$ for clarity.  
A particularly useful class of special expectation values are the
*moments*. The $n$-th moment of the PDF $p$ is defined as
follows
\begin{equation*}
\langle x^n \rangle \equiv \int\! x^n p(x)\,dx
\end{equation*}


<!-- !split -->
### Stochastic variables and the main concepts, mean values
The zero-th moment $\langle 1\rangle$ is just the normalization condition of
$p$. The first moment, $\langle x\rangle$, is called the *mean* of $p$
and often denoted by the letter $\mu$
\begin{equation*}
\langle x\rangle  \equiv \mu = \int x p(x)dx,
\end{equation*}
for a continuous distribution and 
\begin{equation*}
\langle x\rangle  \equiv \mu = \sum_{i=1}^N x_i p(x_i),
\end{equation*}
for a discrete distribution. 
Qualitatively it represents the centroid or the average value of the
PDF and is therefore simply called the expectation value of $p(x)$.



<!-- !split -->
### Mean, median, average
The values of the **mode**, **mean**, **median** can all be used as point estimates for the "probable" value of $x$. For some pdfs, they will all be the same.



<!-- <img src="fig/BayesianBasics/pdfs.png" width=800><p><em>The 68/95 percent probability regions are shown in dark/light shading. When applied to Bayesian posteriors, these are known as credible intervals or DoBs (degree of belief intervals) or Bayesian confidence intervals. The horizontal extent on the $x$-axis translates into the vertical extent of the error bar or error band for $x$.</em></p> -->
![<p><em>The 68/95 percent probability regions are shown in dark/light shading. When applied to Bayesian posteriors, these are known as credible intervals or DoBs (degree of belief intervals) or Bayesian confidence intervals. The horizontal extent on the $x$-axis translates into the vertical extent of the error bar or error band for $x$.</em></p>](./figs/pdfs.png)

<!-- !split -->
### Stochastic variables and the main concepts, central moments, the variance

A special version of the moments is the set of *central moments*, the n-th central moment defined as
\begin{equation*}
\langle (x-\langle x\rangle )^n\rangle  \equiv \int\! (x-\langle x\rangle)^n p(x)\,dx
\end{equation*}
The zero-th and first central moments are both trivial, equal $1$ and
$0$, respectively. But the second central moment, known as the
*variance* of $p$, is of particular interest. For the stochastic
variable $X$, the variance is denoted as $\sigma^2_X$ or $\mathrm{Var}(X)$
\begin{align*}
\sigma^2_X &=\mathrm{Var}(X) =  \langle (x-\langle x\rangle)^2\rangle  =
\int (x-\langle x\rangle)^2 p(x)dx\\
& =  \int\left(x^2 - 2 x \langle x\rangle^{2} +\langle x\rangle^2\right)p(x)dx\\
& =  \langle x^2\rangle - 2 \langle x\rangle\langle x\rangle + \langle x\rangle^2\\
& =  \langle x^2 \rangle - \langle x\rangle^2
\end{align*}
The square root of the variance, $\sigma =\sqrt{\langle (x-\langle x\rangle)^2\rangle}$ is called the 
**standard deviation** of $p$. It is the RMS (root-mean-square)
value of the deviation of the PDF from its mean value, interpreted
qualitatively as the "spread" of $p$ around its mean.





<!-- !split -->
### Probability Distribution Functions

The following table collects properties of probability distribution functions.
In our notation we reserve the label $p(x)$ for the probability of a certain event,
while $P(x)$ is the cumulative probability. 



|   | Discrete PDF |  Continuous PDF |            
| :--- | :----------- | :-------------- |  
| Domain       | $\left\{x_1, x_2, x_3, \dots, x_N\right\}$ | $[a,b]$ |                
| Probability  | $p(x_i)$                       | $p(x)dx$  |              
| Cumulative   | $P_i=\sum_{l=1}^ip(x_l)$       | $P(x)=\int_a^xp(t)dt$  |        
| Positivity   | $0 \le p(x_i) \le 1$           | $p(x) \ge 0$           |  
| Positivity   | $0 \le P_i \le 1$              | $0 \le P(x) \le 1$     |      
| Monotonic    | $P_i \ge P_j$ if $x_i \ge x_j$ | $P(x_i) \ge P(x_j)$ if $x_i \ge x_j$ | 
| Normalization | $P_N=1$                       | $P(b)=1$ |




<!-- !split -->
### Quick introduction to  `scipy.stats`
If you google `scipy.stats`, you'll likely get the manual page as the first hit: [https://docs.scipy.org/doc/scipy/reference/stats.html](https://docs.scipy.org/doc/scipy/reference/stats.html). Here you'll find a long list of the continuous and discrete distributions that are available, followed (scroll way down) by many different methods (functions) to extract properties of a distribution (called Summary Statistics) and do many other statistical tasks.

Follow the link for any of the distributions (your choice!) to find its mathematical definition, some examples of how to use it, and a list of methods. Some methods of interest to us here:

 * `mean()` - Mean of the distribution.
 * `median()` - Median of the distribution.
 * `pdf(x)` - Value of the probability density function at x.
 * `rvs(size=numpts)` - generate numpts random values of the pdf.
 * `interval(alpha)` - Endpoints of the range that contains alpha percent of the distribution.
