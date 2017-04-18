function [accuracy, row, col] = runFinal()
clear;
load temp_final;
addpath('../final')
node_num = size(unique(node_A), 1);

S1 = final_N(A, B, H, node_A, node_B,...
    node_num, 0.3, 40, 1);
[M1, ~] = greedy_match(S1);
[row, col] = find(M1 == 1);

row_ = Pa*row;
col_ = Pb*col;
accuracy = sum(row_ == col_) / size(row_, 1);

