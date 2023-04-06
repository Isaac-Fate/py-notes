# Welcome to This Dummy Notebook

Hello, world!

We shall introduce derivatives in Chapter {doc}`differentiation/index`.

```{note}
This is a note.
```

```{caution}
This is a caution.
```

```{admonition} Exercise
This is an exercise!
```

```{admonition} Solution
:class: tip, dropdown
This is a suggested solution...
```


```{math}
:label: eq:1
\begin{align}
    e^{z} = \sum_{n=0}^\infty \frac{z^n}{n!}
\end{align}
```

```{math}
:label: eq:2
\begin{align}
    e^{x} = \sum_{n=0}^\infty \frac{x^n}{n!} \quad \text{by equation {ref}`eq:1`}
\end{align}
```


```{math}
\begin{align}
    e^{z} = \sum_{n=0}^\infty \frac{z^n}{n!}
\end{align}
```

The definition of $e^z$, see {eq}`eq:1`

See {prf:ref}`alg:1` for details.

```{prf:algorithm}
:label: alg:1
:nonumber:

- [1] Do A
- [2] `for` $i = 1, \ldots, n$

    - 📎 this is a comment
    - [3] do something w.r.t. $i$
    - `for` $j = 1, \ldots, i-1$
        - update $a_{ij}$
            - abababa
                - $\mathbf{p} \gets r_{ij} \mathbf{e}_i$
                - 
                ```{math} 
                \sum_{j=1}^{\infty} a_{ij} = 0 
                ```
            - ababa
    - `end`
- `end`
- Do B

```

```{prf:algorithm}
:label: alg:2
:nonumber:

Do A

`for` $i = 1, \ldots, n$

    do something w.r.t. $i$

    `for` $j = 1, \ldots, i-1$

        update $a_{ij}$

    `end`

`end`

Do nothing

```




```{tableofcontents}
```

## References