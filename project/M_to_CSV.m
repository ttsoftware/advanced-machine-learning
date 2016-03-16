function [] = M_to_CSV( inputDir, outputDir )
    filesList = TOOL_ExtractFilesNames(inputDir, cell(0,0), 1);
    for i=1:length(filesList)
        file = filesList(i,1);
        [pathstr,name,ext] = fileparts(file{1});
        if strcmp(ext,'.mat')
            load(file{1});
            fullPath = fullfile(outputDir, strcat(name, '.csv'));
            csvwrite(fullPath, data);
        end
    end
end

