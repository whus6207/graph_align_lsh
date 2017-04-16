function accuracy = runIsoRank()
clear;
load temp_iso;
addpath('../netalign/matlab')
[S,w,li,lj] = netalign_setup(A,B,L);
x = isorank(S,w,0,1,li,lj);
[ma, mb, ~, ~, ~] = mwmround(x,S,w,li,lj);

[m, n] = size(Pa);
M1 = zeros(m, 1);
for i = ma
	M1(i) = 1;
end
M2 = -1 * ones(m, 1);
for j = mb
	M2(j) = 1;
end
M1 = Pa*M1;
M2 = Pb*M2;
accuracy = sum(M1 == M2) / m;
