usepackage("fourier");
defaultpen(font("T1","fut\textfamilyextension","m","n"));
if(!settings.multipleView)
 settings.batchView=false;
settings.tex="pdflatex";
settings.inlinetex=true;
deletepreamble();
defaultfilename="LaTeX-Vorlage-1";
if(settings.render < 0) settings.render=4;
settings.inlineimage=true;
settings.embed=true;
settings.outformat="";
settings.toolbar=false;
viewportmargin=(2,2);

include graph;
size(1inch);
filldraw(circle((0,0),1),yellow,black);
fill(circle((-.3,.4),.1),black);
fill(circle((.3,.4),.1),black);
draw(arc((0,0),.5,-140,-40));
viewportsize=(299.56693pt,0);
