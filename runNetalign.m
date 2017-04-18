function [accuracy, ma, mb] = runNetalign()
clear;
load temp;
addpath('../netalign/matlab')
[S,w,li,lj] = netalign_setup(A,B,L);
x = netalignbp(S,w,0,1,li,lj);
[ma, mb, ~, ~, ~] = mwmround(x,S,w,li,lj);

[m, n] = size(Pa);
M1 = zeros(n, 1);
for i = ma
	M1(i) = 1;
end

[m_, n_] = size(Pb);
M2 = zeros(n_, 1);
for j = mb
	M2(j) = 1;
end
M1 = Pa*M1;
M2 = Pb*M2;
accuracy = sum(M1.*M2) / m;
