function savemfcc2()
        videos_root = '../Dataset/'
	files = dir('../Dataset/*.mp4');

        %f = fopen(filelist_fname,'rt');
        opt.fs = 16000;
        opt.Tw = 25;
        opt.Ts = 10;
        opt.alpha = 0.97;
        opt.R = [300 3700];
        opt.M = 40;
        opt.C = 13;
        opt.L = 22;

        for i = 1:length(files)
          thisline = files(i).name(1:end-4)
            video_filename = sprintf('%s%s.wav', videos_root, thisline)

            system(sprintf('ffmpeg -loglevel panic -y -threads 1 -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -f wav temp.wav', video_filename))
            [Speech, fs] = audioread('temp.wav');
            [length_of_speech, channel] = size(Speech);
            if channel == 2
                Speech = (Speech(:, 1));
            end
        
            [ MFCCs, ~, ~ ] = runmfcc( Speech, opt );
            mfccs = MFCCs(2:end, :);
			disp (size(mfccs))
            save(sprintf('%s%s.mat', videos_root, thisline), 'mfccs');

        end


