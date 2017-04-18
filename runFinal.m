function [accuracy, row, col] = runFinal()
clear;
load temp_final;
addpath('../final')
node_num = size(unique(node_A), 1);

S1 = final_N(A, B, H, node_A, node_B,...
    node_num, 0.3, 40, 1);
[M1, ~] = greedy_match(S1);
[row, col] = find(M1 == 1);
%row_ = Pa*row;
%col_ = Pb*col;
cnt = 0;
for i = 1:length(row)
	if Pb(row(i), col(i)) == 1
		cnt = cnt + 1;
	end
end
accuracy = cnt / size(row, 1);

