function accuracy = runNetalign()
load temp;
addpath('../netalign/matlab')
[S,w,li,lj] = netalign_setup(A,B,L);
x = netalignbp(S,w,0,1,li,lj);
[ma, mb, ~, ~, ~] = mwmround(x,S,w,li,lj);
accuracy = sum(ma == mb) / size(ma,1);
