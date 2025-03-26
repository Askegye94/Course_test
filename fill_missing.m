% Helper function to fill missing values in a vector
function filled = fill_missing(x)
    filled = x;
    n = length(x);
    for i = 1:n
        if isnan(filled(i))
            % Check for the first non-NaN value later in the vector
            forward_val = [];
            for j = i+1:n
                if ~isnan(filled(j))
                    forward_val = filled(j);
                    break;
                end
            end
            if ~isempty(forward_val)
                filled(i) = forward_val;
            else
                % If not found, check backwards for the most recent non-NaN value
                backward_val = [];
                for j = i-1:-1:1
                    if ~isnan(filled(j))
                        backward_val = filled(j);
                        break;
                    end
                end
                if ~isempty(backward_val)
                    filled(i) = backward_val;
                else
                    % If no valid value is found, set to 0
                    filled(i) = 0;
                end
            end
        end
    end
end