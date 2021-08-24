# Course aim
The course introduces a variety of central algorithms and methods essential for performing scientific data analysis using statistical inference and machine learning. Much emphasis is put on practical applications of Bayesian inference in the natural and engineering sciences, i.e. the ability to quantify the strength of inductive inference from facts (such as experimental data) to propositions such as scientific hypotheses and models.

The course is project-based, and the students will be exposed to fundamental research problems through the various projects, with the aim to reproduce state-of-the-art scientific results. The students will use the Python programming language, with relevant open-source libraries, and will learn to develop and structure computer codes for scientific data analysis projects.

# Teachers

## Course examiner
* _Name_: Christian Forssén
  * _Email_: christian.forssen@chalmers.se
  * _Office_: Department of Physics, Chalmers, room Origo N-6.114 

## Instructors
* _Lecturer_: Christian Forssén
  * _Email_: christian.forssen@chalmers.se
  * _Office_: Department of Physics, Chalmers, room Origo N-6.114 
* _Teaching assistant_: Shahnawaz Ahmed
  * _Email_: shahnawaz.ahmed@chalmers.se
  * _Office_: floor 5, MC2
* _Teaching assistant_: Noemi Bosio
  * _Email_: bosio@chalmers.se
  * _Office_: floor 5, Forskarhuset
* _Teaching assistant_: Isak Svensson
  * _Email_: isak.svensson@chalmers.se
  * _Office_: floor 6, Origo N
* _Teaching assistant_: Oliver Thim
  * _Email_: toliver@chalmers.se
  * _Office_: floor 6, Origo N
  
<!-- !split -->
## About these lecture notes

These lecture notes have been authored by [Christian Forssén](hhttps://www.chalmers.se/en/Staff/Pages/Christian-Forssen.aspx) and are released under a [Creative Commons BY-NC license](https://creativecommons.org/licenses/by-nc/4.0/). The book format is powered by [Jupyter Book](https://jupyterbook.org/).

```{admonition} Open an issue
  If you find a problem or have a suggestion when using this Jupyter Book (on physics, statistics, python, or formatting), from any page go under the github icon <img src="/_images/GitHub-Mark-32px.png" alt="github download icon" width="20px"> at the top-middle-right and select "open issue" (you may want to open in a new tab by *right-clicking* on "open issue"). This will take you to the Issues section of the Github repository for the book. You can either use the title already there or write your own, and then describe in the bigger box your problem or suggestion.
  ```

<!-- ======= Acknowledgements ======= -->
## Acknowledgements

These notes originated from an intensive three-week summer school course taught at the [University of York](https://www.york.ac.uk/) in 2019 by Christian Forssén, Dick Furnstahl, and Daniel Phillips as part of the [TALENT](https://fribtheoryalliance.org/TALENT/) initiative. The original notes and subsequent revisions have been informed by interactions with many colleagues; I am particularly grateful to:

* Prof. Richard Furnstahl, Ohio State University
* Prof. Morten Hjorth-Jensen, Oslo University and Michigan State University
* Prof. Daniel Phillips, Ohio University.
* Dr. Ian Vernon, Durham University
* Dr. Sarah Wesolowski, Salisbury University

The full list of people that have contributed with ideas, discussions, or by generously sharing their knowledge is very long. Rather than inadvertently omitting someone, I simply say thank you to all. More generally, I am truly thankful for being part of an academic environment in which ideas and efforts are shared rather than kept isolated.

The last statement extends to the open-source communities that make so many great computing tools publicly available. In this course we take great advantage of open-source python libraries.  

The development of this course also would not have been possible without the knowledge gained through the study of several excellent textbooks, most of which are listed as recommended course literature. Here is a short list of those references that I have found particularly useful as a physicist learning Bayesian statistics and machine learning:

1. Phil Gregory, *"Bayesian Logical Data Analysis for the Physical Sciences"*, Cambridge University Press (2005).
2. E. T. Jaynes, *"Probability Theory: The Logic of Science"*, Cambridge University Press (2003).
3. David J.C. MacKay, *"Information Theory, Inference, and Learning Algorithms"*, Cambridge University Press (2005).
4. D.S. Sivia, *"Data Analysis : A Bayesian Tutorial"*, Oxford University Press (2006).

## Brief guide to online Jupyter Book features

* A clickable high-level table of contents (TOC) is available in the panel at the left of each page. (You can close this panel with the left arrow at the top-left-middle of the page or open it with the contents icon at the upper left.) 
    ```{admonition} Searching the book
    The "Search this book..." box just below the title in the TOC panel is a great tool.     Try it! (And enhancements are expected in the near future as the Jupyter Book project matures.)
    ```
* For each section that has subsections, a clickable table of contents appears in the rightmost panel.
* On pages that are not generated from Jupyter notebooks, the three icons at the top-middle-right will put you into full-screen mode; take you to the github repository for the book or let you open an issue (see the top of this page); or show you the markdown source (.md) of the page or generate a pdf version of the page.

