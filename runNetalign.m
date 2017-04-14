function accuracy = runNetalign()
load temp;
addpath('../netalign/matlab')
[S,w,li,lj] = netalign_setup(A,B,L);
x = netalignbp(S,w,0,1,li,lj);
[ma, mb, ~, ~, ~] = mwmround(x,S,w,li,lj);
size(Pa)
size(ma)
size(Pb)
size(mb)
ma = transpose(Pa)*ma;
mb = transpose(Pb)*mb;
accuracy = sum(ma == mb) / size(ma,1);
