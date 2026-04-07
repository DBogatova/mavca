function mat_to_csv(matFile)
% Convert MAT file to:
%   <base>_trigger.csv
%   <base>_accel.csv
%   <base>_trigger_info.txt
%
% Alignment rule:
%   Time zero = first rising edge of AndorXylaTrigger (0 -> 1)

    if nargin < 1
        error('Usage: mat_to_csv(''Run008_t1.mat'')');
    end

    S = load(matFile);

    if ~isfield(S, 'data')
        error('MAT file does not contain field "data".');
    end

    ai = S.data.ai;
    di = S.data.di;

    % Sampling rate
    Fs = 1000;  % fallback
    if isfield(S, 'device')
        if isfield(S.device, 'actualRate') && ~isempty(S.device.actualRate)
            Fs = double(S.device.actualRate);
        elseif isfield(S.device, 'rate') && ~isempty(S.device.rate)
            Fs = double(S.device.rate);
        end
    end

    [folder, baseName, ~] = fileparts(matFile);
    if isempty(folder)
        folder = pwd;
    end

    trigger_csv = fullfile(folder, [baseName '_trigger.csv']);
    accel_csv   = fullfile(folder, [baseName '_accel.csv']);
    info_txt    = fullfile(folder, [baseName '_trigger_info.txt']);

    %% ---------------------------
    % Find first rising edge of AndorXylaTrigger
    % ----------------------------
    if ~ismember('AndorXylaTrigger', di.Properties.VariableNames)
        error('AndorXylaTrigger not found in data.di');
    end

    trig = double(di.AndorXylaTrigger(:));

    % Rising edge = sample i where trig(i-1)==0 and trig(i)==1
    rising_idx = find(diff(trig) > 0, 1, 'first') + 1;

    if isempty(rising_idx)
        error('No rising edge found in AndorXylaTrigger.');
    end

    % Zero-based sample index for compatibility with Python indexing
    start_sample = rising_idx - 1;
    start_time_s = start_sample / Fs;

    %% ---------------------------
    % Save trigger CSV
    % ----------------------------
    nTrig = height(di);
    sample = (0:nTrig-1)';
    time_s = sample / Fs;
    aligned_time_s = time_s - start_time_s;

    trigger_tbl = table(sample, time_s, aligned_time_s, ...
        'VariableNames', {'sample', 'time_s', 'aligned_time_s'});

    for k = 1:numel(di.Properties.VariableNames)
        varName = di.Properties.VariableNames{k};
        trigger_tbl.(varName) = double(di.(varName));
    end

    writetable(trigger_tbl, trigger_csv);

    %% ---------------------------
    % Save accelerometer CSV
    % ----------------------------
    required_ai = {'accX','accY','accZ'};
    for k = 1:numel(required_ai)
        if ~ismember(required_ai{k}, ai.Properties.VariableNames)
            error('Missing accelerometer channel: %s', required_ai{k});
        end
    end

    accX = double(ai.accX(:));
    accY = double(ai.accY(:));
    accZ = double(ai.accZ(:));

    nAcc = numel(accX);
    acc_sample = (0:nAcc-1)';
    acc_time_s = acc_sample / Fs;
    acc_aligned_time_s = acc_time_s - start_time_s;

    % Center channels a bit for nicer movement plotting
    accX0 = accX - median(accX, 'omitnan');
    accY0 = accY - median(accY, 'omitnan');
    accZ0 = accZ - median(accZ, 'omitnan');

    accel_mag = sqrt(accX0.^2 + accY0.^2 + accZ0.^2);

    % Keep accel_mag in column 2 because your Python script reads iloc[:,1]
    accel_tbl = table(acc_sample, accel_mag, accX, accY, accZ, ...
        acc_time_s, acc_aligned_time_s, ...
        'VariableNames', {'sample','accel_mag','accX','accY','accZ','time_s','aligned_time_s'});

    writetable(accel_tbl, accel_csv);

    %% ---------------------------
    % Save info TXT
    % ----------------------------
    fid = fopen(info_txt, 'w');
    if fid == -1
        error('Could not write info file.');
    end

    fprintf(fid, 'MAT to CSV conversion\n');
    fprintf(fid, '=====================\n\n');
    fprintf(fid, 'Input file: %s\n', matFile);
    fprintf(fid, 'Sampling rate: %.6f Hz\n\n', Fs);

    fprintf(fid, 'Alignment definition:\n');
    fprintf(fid, 'Time zero = first rising edge of AndorXylaTrigger (0 -> 1)\n\n');

    fprintf(fid, 'AndorXylaTrigger start:\n');
    fprintf(fid, '  sample = %d\n', start_sample);
    fprintf(fid, '  time_s = %.6f\n\n', start_time_s);

    fprintf(fid, 'Output files:\n');
    fprintf(fid, '  %s\n', trigger_csv);
    fprintf(fid, '  %s\n', accel_csv);
    fprintf(fid, '  %s\n', info_txt);

    fclose(fid);

    %% ---------------------------
    % Print summary
    % ----------------------------
    fprintf('\nSaved files:\n');
    fprintf('  %s\n', trigger_csv);
    fprintf('  %s\n', accel_csv);
    fprintf('  %s\n', info_txt);

    fprintf('\nAndorXylaTrigger first rising edge:\n');
    fprintf('  sample = %d\n', start_sample);
    fprintf('  time   = %.6f s\n', start_time_s);
end