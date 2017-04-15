function accuracy = runNetalign()
load temp;
addpath('../netalign/matlab')
[S,w,li,lj] = netalign_setup(A,B,L);
x = netalignbp(S,w,0,1,li,lj);
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
ma = transpose(Pa)*M1;
mb = transpose(Pb)*M2;
accuracy = sum(M1 == M2) / m;
