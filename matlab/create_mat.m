function savemfcc_for_test(input_path, output_path)
        opt.fs = 16000;
        opt.Tw = 25;
        opt.Ts = 10;
        opt.alpha = 0.97;
        opt.R = [300 3700];
        opt.M = 40;
        opt.C = 13;
        opt.L = 22;

        if isempty(strfind(input_path, '.wav'))
            system(sprintf('ffmpeg -loglevel panic -y -threads 1 -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -f wav temp.wav', input_path))

        else
            system(sprintf('ffmpeg -loglevel panic -y -threads 1 -i %s -async 1 -ac 1 -ar 16000 -f wav temp.wav', input_path))
        end

        [Speech, fs] = audioread('temp.wav');
        [length_of_speech, channel] = size(Speech);
        if channel == 2
            Speech = (Speech(:, 1));
        end
        
        [ MFCCs, ~, ~ ] = runmfcc( Speech, opt );
        mfccs = MFCCs(2:end, :);

        save(output_path, 'mfccs');
