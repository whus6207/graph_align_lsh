function runNetalign()
fd = fopen('netalignScore.txt','w');
files = dir('*.mat');
for file = files'
    load(file.name)
    score = executeNetalign(A, B, L);
    fprintf(fd,'%s %f\n',file.name,score);
end
fclose(fd);
end