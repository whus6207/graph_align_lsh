function [A, B, L] = loadEdges(path)
file = fopen(strcat(path,'/A.edges'),'r');
A = fscanf(file,'%d %d',[2 Inf]);
fclose(file);

file = fopen(strcat(path,'/B.edges'),'r');
B = fscanf(file,'%d %d',[2 Inf]);
fclose(file);

file = fopen('./metadata/sim_mat.txt','r');
L = fscanf(file,'%f',[max(max(A))+1 max(max(B))+1]);
fclose(file);

A = sparse(transpose(A(1,:))+1, transpose(A(2,:))+1, ones(1,size(A, 2)), max(max(A))+1, max(max(A))+1);
B = sparse(transpose(B(1,:))+1, transpose(B(2,:))+1, ones(1,size(B, 2)), max(max(B))+1, max(max(B))+1);
L = sparse(L);

