# SVM

## Separating Hyperplanes

Hyperplane is an affine subspace of dim D-1. We have different aproaches for this:
- Geometric View
- Loss function View
- Dual Problem

Let example $x \in R^D$ be an element of the data space. Consider,
$$f: R^D \rightarrow R \\ x \rightarrow f(x) := <w, x> + b$$
parametrized by $w \in R^D$ and $b \in R$

imgs

where vector w is a vector normal to the hyperplane and b the interpect.
We can prove that by choosing any two examples $x_a$ and $x_b$ on the hyperplane and showing that the vector between them is orthogonal to w.

$$f(x_a) - f(x_b) = <w,x_a> + b - (<w, x_b> + b) \\ = <w, x_a - x_b>$$

We chose $x_a$ and $x_b, f(x_a) = 0$ and $f(x_b) = 0$ and hence $<w, x_a - x_b> = 0$.

img

We are defining a hyperplane and a direction, positive and negative side of the hyperplane. To classify $x_{test}$ we calculate $f(x_{test})$ and classify it as +1 if $f(x_{test}) \ge 0$ and -1 otherwise. 
> Thinking geometrically, the positive example lie above the hyperplane and the negative below it.

when training, we want to ensure that positive labels are on the positive side of the hyperplane, i.e.,  
$<w, xn> + b \ge 0$