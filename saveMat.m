function saveMat(A, B, L, s)
A = sparse(A);
B = sparse(B);
L = sparse(L);
save(s,'A','B','L');
