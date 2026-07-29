$pdf_mode = 1;
# No -shell-escape: aligns local/CI with arXiv default pdflatex (no minted/epstopdf).
$pdflatex = 'pdflatex -interaction=nonstopmode -halt-on-error -file-line-error %O %S';
$bibtex_use = 2;
$out_dir = 'build';
$clean_ext = 'synctex.gz acn acr alg aux bbl bcf blg dvi fdb_latexmk fls glg glo gls idx ilg ind ist lof log lol lot nav nlg nlo nls out run.xml snm thm toc vrb xdy';
