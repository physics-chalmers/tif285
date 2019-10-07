#!/bin/sh
# Shell script for TIF285 lecture notes
set -x

function system {
  "$@"
  if [ $? -ne 0 ]; then
    echo "make.sh: unsuccessful command $@"
    echo "abort!"
    exit 1
  fi
}

if [ $# -eq 0 ]; then
echo 'bash make.sh slides1|slides2'
echo 'bash make.sh slides1 web to publish in the ../../web folder'
exit 1
fi

name=$1
publish=$2

rm -f *.tar.gz

opt=''
encoding="--encoding=utf-8"

preprocess -DFORMAT=pdflatex ../newcommands.p.tex > newcommands_keep.tex

html=${name}-reveal
system doconce format html $name --pygments_html_style=perldoc --keep_pygments_html_bg --html_links_in_new_window --html_output=$html $opt
system doconce slides_html $html reveal --html_slide_theme=beige

# Plain HTML documentsls

#--html=${name}-solarized
#--system doconce format html $name --pygments_html_style=perldoc --html_style=solarized3 --html_links_in_new_window --html_output=$html $opt
#--system doconce slides_html $html doconce --nav_button=text method=hrule

#html=${name}
#system doconce format html $name --pygments_html_style=default --html_style=bloodish #--html_links_in_new_window --html_output=$html $opt
#system doconce split_html $html.html
## Remove top navigation in all parts
#doconce subst -s '<!-- begin top navigation.+?end top navigation -->' '' #${name}-plain.html ._${name}*.html



# Ordinary plain LaTeX document
system doconce format pdflatex $name $opt $encoding --minted_latex_style=trac --latex_admon=mdfbox --latex_admon_color=1,1,1 --latex_table_format=left --latex_admon_title_no_period --latex_code_style=default:lst[style=blue1]@pypro:lst[style=blue1bar]@dat:lst[style=gray]@sys:vrb[frame=lines,label=\\fbox{{\tiny Terminal}},framesep=2.5mm,framerule=0.7pt]  

#system doconce ptex2tex $name envir=minted
# Add special packages
doconce replace '% insert custom LaTeX commands...' '\usepackage[swedish]{babel}' $name.tex
doconce subst 'frametitlebackgroundcolor=.*?,' 'frametitlebackgroundcolor=blue!5,' 

#rm -rf $name.aux $name.ind $name.idx $name.bbl $name.toc $name.loe

system latexmk -pdf -shell-escape $name
#pdflatex -shell-escape $name
#pdflatex -shell-escape $name

mv -f $name.pdf ${name}-print.pdf
cp $name.tex ${name}-plain-print.tex

# IPython notebook
system doconce format ipynb $name $opt

# Bootstrap style
html=${name}-bs
system doconce format html $name --html_style=bootstrap --pygments_html_style=default --html_admon=bootstrap_panel --html_output=$html $opt
#system doconce split_html $html.html --pagination --nav_button=bottom

# Publish
if [ "$2" = "web" ]; then
dest=../../web
elif [ "$2" = "pub" ]; then
dest=../../pub
else
exit 0
fi
if [ ! -d $dest/$name ]; then
mkdir $dest/$name
mkdir $dest/$name/pdf
mkdir $dest/$name/html
mkdir $dest/$name/ipynb
fi
cp ${name}*.pdf $dest/$name/pdf/.
cp -r ${name}-bs.html $dest/$name/html
#cp -r ${name}*.html ._${name}*.html reveal.js $dest/$name/html

for format in "html" "ipynb"; do
# Figures: cannot just copy link, need to physically copy the files
if [ -d fig ]; then
if [ ! -d $dest/$name/$format/fig ]; then
mkdir $dest/$name/$format/fig
fi
cp -r fig/* $dest/$name/$format/fig/.
fi
# Figures are evne better stored in fig-$name
if [ -d fig-${name} ]; then
if [ ! -d $dest/$name/$format/fig-${name} ]; then
mkdir $dest/$name/$format/fig-${name}
fi
cp -r fig-${name}/* $dest/$name/$format/fig-${name}/.
fi
# The same with data files
if [ -d data ]; then
if [ ! -d $dest/$name/$format/data ]; then
mkdir $dest/$name/$format/data
fi
cp -r data/* $dest/$name/$format/data/.
fi
if [ -d data-${name} ]; then
if [ ! -d $dest/$name/$format/data-${name} ]; then
mkdir $dest/$name/$format/data-${name}
fi
cp -r data-${name}/* $dest/$name/$format/data-${name}/.
fi
done

cp ${name}.ipynb $dest/$name/ipynb/.
# Possible demo notebook
ipynb_demo=demo-${name}.ipynb
if [ -f ${ipynb_demo} ]; then
cp ${ipynb_demo} $dest/$name/ipynb/.
fi

#ipynb_tarfile=ipynb-${name}-src.tar.gz
#if [ ! -f ${ipynb_tarfile} ]; then
#cat > README.txt <<EOF
#This IPython notebook ${name}.ipynb does not require any additional
#programs.
#EOF
#tar czf ${ipynb_tarfile} README.txt
#fi
#cp ${ipynb_tarfile} $dest/$name/ipynb/.

