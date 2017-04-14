clear;
function accuracy = runFinal()
load(temp_final);
addpath('../final')
S1 = final_NE(A, B, H, node_A, node_B,...
    A, B,...
    1, 1, 0.6, 40, 0);
[M1, ~] = greedy_match(S1);
[row, col] = find(M1 == 1);

row = Pa*row;
col = Pb*col;
// ground_sort = sortrows(ground_truth, 2);
// accuracy = sum(row == ground_sort(:,1))/length(row)
accuracy = sum(row == col) / size(row, 1)