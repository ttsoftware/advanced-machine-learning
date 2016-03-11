function filesList = TOOL_ExtractFilesNames(rootFolder, filesList, stack)
    stack = stack+1;
    liste = dir(rootFolder);
    for i = 1:length(liste)
        if ( ~strcmp(liste(i).name,'.') & ~strcmp(liste(i).name,'..') )
            if ( liste(i).isdir == 1 )
                filesList = TOOL_ExtractFilesNames(fullfile(rootFolder,liste(i).name), filesList, stack);
            else
                filesList{end+1,1} = fullfile(rootFolder,liste(i).name);
            end
        end
    end
end



