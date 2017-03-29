function score = executeNetalign(A, B, L)
%[A, B] = loadEdges(path);
addpath('./netalign/matlab')
[S,w,li,lj] = netalign_setup(A,B,L);
x = netalignbp(S,w,0,1,li,lj);
[ma, mb, ~, ~, ~] = mwmround(x,S,w,li,lj);
diff = (ma - mb)~=0;
score = sum(diff) / size(diff,1);