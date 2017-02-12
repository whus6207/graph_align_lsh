function [ma, mb, mi, overlap, weight] = executeNetalign(A, B, L)
%[A, B] = loadEdges(path);
addpath('./netalign/matlab')
[S,w,li,lj] = netalign_setup(A,B,L);
x = netalignbp(S,w,0,1,li,lj);
[ma, mb, mi, overlap, weight] = mwmround(x,S,w,li,lj);
