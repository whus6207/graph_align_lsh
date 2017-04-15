function accuracy = runFinal()
clear;
load temp_final;
addpath('../final')
S1 = final_NE(A, B, H, node_A, node_B,...
    A, B,...
    1, 1, 0.6, 40, 0);
[M1, ~] = greedy_match(S1);
[row, col] = find(M1 == 1);

row = Pa*row;
col = Pb*col;
accuracy = sum(row == col) / size(row, 1);