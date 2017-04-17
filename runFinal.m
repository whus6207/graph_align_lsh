function [accuracy, row, col] = runFinal()
clear;
load temp_final;
addpath('../final')
S1 = final_NE(A, B, H, node_A, node_B,...
    A, B,...
    1, 1, 0.6, 40, 1);
[M1, ~] = greedy_match(S1);
[row, col] = find(M1 == 1);

row_ = Pa*row;
col_ = Pb*col;
accuracy = sum(row_ == col_) / size(row_, 1);

