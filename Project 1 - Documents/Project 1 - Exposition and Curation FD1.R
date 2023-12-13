---
  title: "Project 1: Data for Question 7 and Exposition and Curation"
author: "Robert Williams"
date: "`r Sys.Date()`"
output:
  html_document:
  code_folding: hide
number_sections: false
toc: yes
toc_depth: 3
toc_float: yes
pdf_document:
  toc: yes
toc_depth: '3'
---
  
  ```{r init, include=F}
# The package "ezids" (EZ Intro to Data Science) includes some helper functions we developed for the course. 
# Some of the frequently used functions are loadPkg(), xkabledply(), xkablesummary(), uzscale(), etc.
# You will need to install it (once) from GitHub.
# library(devtools)
# devtools::install_github("physicsland/ezids")
# Then load the package in your R session.
library(ezids)
```


```{r setup, include=FALSE}
# Some of common RMD options (and the defaults) are: 
# include=T, eval=T, echo=T, results='hide'/'asis'/'markup',..., collapse=F, warning=T, message=T, error=T, cache=T, fig.width=6, fig.height=4, fig.dim=c(6,4) #inches, fig.align='left'/'center','right', 
knitr::opts_chunk$set(warning = F, message = F)
# Can globally set option for number display format.
options(scientific=T, digits = 3) 
# options(scipen=9, digits = 3) 
```

```{r}
# 1. Do not provide answers/comments inside code blocks (like here) -- those are notes between coders/self and will be ignored for grading. 
# 2. Make sure your knitr options are set to include all results/code to be graded in the final document.
# 3. All charts/graphs/tables should have appropriate titles/labels/captions. 
# 4. Compose your answers using inline R code instead of using the code-block output as much as you can. 
# 5. Your grade is also determined by the style. Even if you answer everything correctly, but the .html does not look appealing, you will not get full credit. Pay attention to the details that we mentioned in class/homework and in previous sample .Rmd files. For example, how to use #, ##, ###, ..., bold face, italics, inline codes, tables, ..., {results = "asis"}, use of colors in plots/ggplots, and so forth.
```
