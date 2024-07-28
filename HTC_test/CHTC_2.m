% Read the file
f = fopen('grid_Visc_logD_AlvaDesc_2_1.py','r');
content = textscan(f,'%s','Delimiter','\n','Whitespace','');
fclose(f);

for i = 2:10
    % Change the 46th line
    content{1}{45} = sprintf('for num_state in range(%d, %d):', 50*i-49, 50*i+1);
    content{1}{103} = sprintf('                GS_result.to_csv("result_Visc_logD_AlvaDesc_2(%d).csv")', i);
    % Write to a new file
    filename = sprintf('grid_Visc_logD_AlvaDesc_2_%d.py', i);  % Include 'i' in the filename
    f = fopen(filename,'w');
    for j=1:length(content{1})
        fprintf(f,'%s\n',content{1}{j});
    end
    fclose(f);
end

for i = 1:10
    fprintf('docker cp /home/tyue4/Visc/grid_Visc_logABDC_1/grid_Visc_logD_AlvaDesc_2_%d.py 078a4eb9da49:/Visc_ABCD/grid_Visc_logD_AlvaDesc_2_%d.py\n',i,i);
end

for i = 1:10
    % 新建文件夹
    dirName = sprintf('%d', i);
    if ~exist(dirName, 'dir')
       mkdir(dirName)
    end
    
    content{1} = sprintf('python /Visc_ABCD/grid_Visc_logD_AlvaDesc_2_%d.py', i);
    content{2} = sprintf('tar -czf result.tar.gz /Visc_ABCD/result_Visc_logD_AlvaDesc_2(%d).csv', i);
    
    % 在新建的文件夹内创建新的.sh文件
    filename = sprintf('%s/moses_ytl_1.sh', dirName);  % 将 'i' 包含在文件名中
    f = fopen(filename,'w');
    if f == -1
        error('无法打开文件 %s 用于写入', filename);
    end
    for j=1:length(content)
        fprintf(f,'%s\n',content{j});
    end
    fclose(f);
end

for i = 1:10
    copyfile('submit.sub', sprintf('%d/submit.sub', i));
end

