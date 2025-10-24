__prepro_errors__ = ["swot_reach_file_not_found", #1
                     "dataset_without_valid_observation", #2
                     "reach_without_valid_observation", #3
                     "NaT_times", #4
                     "cycles_not_in_increasing_order"] #5
__run_errors__ = ["no_valid_observation_profile", #101
                  "all_nan_Qlf_for_some_profiles", #102
                  "null_cprior_or_cpost", #103
                  "nan_cprior_or_cpost"] #104

__critical_errors__ = [1, 4, 5, 101, 102, 103]

__error_codes__ = {err: index+1 for index, err in enumerate(__prepro_errors__)}
__error_codes__.update({err: index+101 for index, err in enumerate(__run_errors__)})

__error_strings__ = {__error_codes__[code]: code for code in __error_codes__}

def error_string_from_code(error_code: int):
    if error_code in __error_strings__:
        return __error_strings__[error_code]
    else:
        return "Unknown error code: %i" % error_code

def error_code_from_string(error_string: int):
    if error_string in __error_codes__:
        return __error_codes__[error_string]
    else:
        return "Unknown error string: %i" % error_string
