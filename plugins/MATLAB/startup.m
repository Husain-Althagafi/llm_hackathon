globalChemEnginePlugin = true;

try
fileName = "_.py";
outputFormat = "o.io";

% pyrun("print('test'") % built-in matlab python interface

while globalChemEnginePlugin

userInput = input('>> ', 's');
system(sprintf('start /B /wait python %s "%s"', fileName, userInput)); % start /B is windows platform specific

f = fopen(sprintf("%s", outputFormat), 'r');
x = fread(f, '*char');
f = fclose(f);

system(sprintf("del %s",outputFormat));

switch(true)
    case ~isempty(x)
        disp(x)
        continue;

    otherwise
        try
            evalin('base', userInput) % eval matlab input

        catch err
            fprintf(2, '%s\n', err.message);
        end
        
        continue;
end
end

catch fileError
    fprintf(2,"%s\n", fileError.message);
end