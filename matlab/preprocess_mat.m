function savemfcc(filelist_fname, videos_root)
        f = fopen(filelist_fname,'rt');
        opt.fs = 16000;
        opt.Tw = 25;
        opt.Ts = 10;
        opt.alpha = 0.97;
        opt.R = [300 3700];
        opt.M = 40;
        opt.C = 13;
        opt.L = 22;

        while true
          thisline = fgetl(f);
          if ~ischar(thisline); break; end  %end of file

            video_filename = sprintf('%s%s.mp4', videos_root, thisline)

            system(sprintf('ffmpeg -loglevel panic -y -threads 1 -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -f wav temp.wav', video_filename))
            [Speech, fs] = audioread('temp.wav');
            [length_of_speech, channel] = size(Speech);
            if channel == 2
                Speech = (Speech(:, 1));
            end
        
            [ MFCCs, ~, ~ ] = runmfcc( Speech, opt );
            mfccs = MFCCs(2:end, :);

            save(sprintf('%s%s.mat', videos_root, thisline), 'mfccs');

        end
        fclose(f);
        
        %{
num_bins = floor(length_of_speech / fs * 25);
        for l = 2:num_bins - 4
            save_mfcc20 = mfccs(:, 4 * l -7  : 4 * l + 19 -7);

            f2 = fopen(fullfile(save_dir, [num2str(l), '.bin']), 'wb');
            fwrite(f2, save_mfcc20, 'double');
            fclose(f2);                    
        end
%}

