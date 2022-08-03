# Introduction

These are the lecture notes for the master-level course "Learning from data" (TIF285/FYM285) that is taught at Chalmers University of Technology and Göteborg University. The accompanying jupyter notebooks can be found in [this git repository](https://github.com/physics-chalmers/tif285).

## Course aim
The course introduces a variety of central algorithms and methods essential for performing scientific data analysis using statistical inference and machine learning. Much emphasis is put on practical applications of Bayesian inference in the natural and engineering sciences, i.e. the ability to quantify the strength of inductive inference from facts (such as experimental data) to propositions such as scientific hypotheses and models.

The course is project-based, and students will be exposed to fundamental research problems through the course projects that aim to reproduce state-of-the-art scientific results. Students will use the Python programming language, with relevant open-source libraries, and will learn to develop and structure computer codes for scientific data analysis projects.

<!-- !split -->
## About these lecture notes

These lecture notes have been authored by [Christian Forssén](https://www.chalmers.se/en/Staff/Pages/Christian-Forssen.aspx) and are released under a [Creative Commons BY-NC license](https://creativecommons.org/licenses/by-nc/4.0/). The book format is powered by [Jupyter Book](https://jupyterbook.org/).

```{admonition} Open an issue
  If you find a problem or have a suggestion when using this Jupyter Book (on physics, statistics, python, or formatting), from any page go under the github icon ![github download icon](./figs/GitHub-Mark-32px.png) at the top-middle-right and select "open issue" (you may want to open in a new tab by *right-clicking* on "open issue"). This will take you to the Issues section of the Github repository for the book. You can either use the title already there or write your own, and then describe in the bigger box your problem or suggestion.
  ```
  
## Brief guide to online Jupyter Book features

* A clickable high-level table of contents (TOC) is available in the panel at the left of each page. (You can close this panel with the left arrow at the top-left-middle of the page or open it with the contents icon at the upper left.) 
    ```{admonition} Searching the book
    The "Search this book..." box just below the title in the TOC panel is a great tool.     Try it! (And enhancements are expected in the near future as the Jupyter Book project matures.)
    ```
* For each section that has subsections, a clickable table of contents appears in the rightmost panel.
* On pages that are not generated from Jupyter notebooks, the three icons at the top-middle-right will put you into full-screen mode; take you to the github repository for the book or let you open an issue (see the top of this page); or show you the markdown source (.md) of the page or generate a pdf version of the page.

<!-- ======= Acknowledgements ======= -->
## Acknowledgements

These notes originated from an intensive three-week summer school course taught at the [University of York](https://www.york.ac.uk/) in 2019 by Christian Forssén, Dick Furnstahl, and Daniel Phillips as part of the [TALENT](https://fribtheoryalliance.org/TALENT/) initiative. The original notes and subsequent revisions have been informed by interactions with many colleagues; I am particularly grateful to:

* Dr. Andreas Ekström, Chalmers University of Technology
* Prof. Richard Furnstahl, Ohio State University
* Prof. Morten Hjorth-Jensen, Oslo University and Michigan State University
* Prof. Daniel Phillips, Ohio University
* Prof. Ian Vernon, Durham University
* Dr. Sarah Wesolowski, University of Pennsylvania

The full list of people that have contributed with ideas, discussions, or by generously sharing their knowledge is very long. Rather than inadvertently omitting someone, I simply say thank you to all. More generally, I am truly thankful for being part of an academic environment in which ideas and efforts are shared rather than kept isolated.

The last statement extends to the open-source communities through which great computing tools are made publicly available. In this course we take great advantage of open-source python libraries.  

The development of this course would not have been possible without the knowledge gained through the study of several excellent textbooks, most of which are listed as recommended course literature. Here is a short list of those references that I have found particularly useful as a physicist learning Bayesian statistics and the fundamentals of machine learning:

1. Phil Gregory, *"Bayesian Logical Data Analysis for the Physical Sciences"*, Cambridge University Press (2005).
2. E. T. Jaynes, *"Probability Theory: The Logic of Science"*, Cambridge University Press (2003).
3. David J.C. MacKay, *"Information Theory, Inference, and Learning Algorithms"*, Cambridge University Press (2005).
4. D.S. Sivia, *"Data Analysis : A Bayesian Tutorial"*, Oxford University Press (2006).


