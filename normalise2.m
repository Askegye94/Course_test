
function y=normalize(VEK, interv)
% Syntax: y=normalize(VEK, interv) - normalisiert den Vektor VEK in Schritte der L?nge 'interv' ausgehend von seiner urspr?nglichen
% "Zeitachse" auf 100% in %-Schritten, wie durch "interv" angegeben. "VEK"
% sind die zu normalisierenden Daten, "interv" gibt die gew?nschte
% Schrittweite der Normalisierung an.
t_mod=0:100/(length(VEK)-1):100;
t_nor=0:interv:100;
y=interp1(t_mod, VEK, t_nor,'pchip');