from datetime import timedelta

import re # Needed for parsing node IDs
import itertools # Needed for parsing node IDs

def print_and_log(logger, message, sep=None):
    if sep: # message is a list of strings
        message_string = ''
        for msg in message:
            message_string += msg + sep
        logger.info(message_string)
        print(message_string)
    else:
        print(message)
        logger.info(message)

def convert_nodelist_to_node_nums(nid_str, system="kestrel"):
    """
    Converts a string of Node IDs to a list of Node IDs. This is necessary
    because the Node IDs are listed in the Slurm config file with numeric range specification,
    and this must be parsed into a list of all Node IDs specified.
    """
    if system == 'kestrel':
        # Matching strings like: 
        # x3112c[0]s[5,9,13,17,21,25]b[0]n[0]
        # x[3100-3105,3110-3111,3113]c[0]s[5,9,13,17,21,25,29,33,37,41]b[0]n[0]
        pattern = re.compile(r'x(\[[^\]]+\]|\d+)c(\[[^\]]+\]|\d+)s(\[[^\]]+\]|\d+)b(\[[^\]]+\]|\d+)n(\[[^\]]+\]|\d+)')

        # Returns a list of 4-tuples: [(x part, c part, b part, n part), ... ,(x part, c part, b part, n part)]
        # e.g.: for this nid = 'x[3100-3105,3110-3111,3113]c[0]s[5,9,13,17,21,25,29,33,37,41]b[0]n[0],x3112c[0]s[5,9,13,17,21,25]b[0]n[0]'
        # We get:
        #    ('[3100-3105,3110-3111,3113]', '[0]', '[5,9,13,17,21,25,29,33,37,41]', '[0]', '[0]')
        #    ('3112', '[0]', '[5,9,13,17,21,25]', '[0]', '[0]')
        matches = pattern.findall(nid_str)

        all_node_ids = []

        for match in matches:
            expanded_parts = list()

            # Expand the part if necessary to get all possible numeric values for the part
            # e.g. [3100-3105,3110-3111,3113] expands to 3100, 3101, 3102, 3103, 3104, 3105, 3110, 3111, 3113
            for part in match:
                if '[' in part:
                    expanded_part = expand_ranges(part.strip('[]'))
                else:
                    expanded_part = [int(part)]
                expanded_parts.append(expanded_part)

            # Get all combinations of all parts to get a list of all Node IDs defined by this match
            for x, c, s, b, n in itertools.product(*expanded_parts):
                node_id = f'x{x}c{c}s{s}b{b}n{n}'
                all_node_ids.append(node_id)

        # Return the entire list of all Node IDs for all matches
        return all_node_ids
    else:
        # This was the previous implementation, I assume for Lumi
        if nid_str == "dummy":
            return -1
        node_nums = []
        
        entry_slice_points = [-1]
        in_brackets = False

        for i_char, char in enumerate(nid_str):
            if char == "[":#]
                in_brackets = True
            elif char == "]":
                in_brackets = False
            elif not in_brackets and char == ",":
                entry_slice_points.append(i_char)

        entry_slice_points.append(None)

        for slice_l, slice_r in zip(entry_slice_points[:-1], entry_slice_points[1:]):
            nid_str_entry = nid_str[slice_l+1:slice_r]
            
            # name001
            if "[" not in nid_str_entry:#]
                node_nums.append(nid_str_entry)
                continue

            if nid_str_entry.count("[") >= 2:#]
                raise NotImplementedError(
                    "Mulitple numeric ranges like {} not implemented".format(nid_str)
                )

            # name00[1-4,10,12,13-20]
            nid_prefix = nid_str_entry.split("[")[0]#]
            nid_suffix_str = nid_str_entry.strip("]").split("[")[1]#]

            for nid_suffix_entry in nid_suffix_str.split(","):
                if "-" not in nid_suffix_entry:
                    node_nums.append(nid_prefix + nid_suffix_entry)
                    continue

                nid_suffix_range = nid_suffix_entry.split("-")
                digits = len(nid_suffix_range[0]) if nid_suffix_range[0].startswith("0") else None

                for node_num in range(int(nid_suffix_range[0]), int(nid_suffix_range[1]) + 1):
                    node_num = str(node_num)

                    if digits is not None:
                        while len(node_num) < digits:
                            node_num = "0" + node_num

                    node_nums.append(nid_prefix + node_num)
        return node_nums


def expand_ranges(s):
    """
    Expands a range of numeric values into a list of numeric values
    defined by the range.
    
    Example input: 3100-3105
    Example output: [3100, 3101, 3102, 3103, 3104, 3105]
    """
    parts = []
    for part in s.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            parts.extend(range(start, end + 1))
        else:
            parts.append(int(part))
    return parts


def get_sbatch_cli_arg(submit_line, long="", short=""):
    """
    Get command line arguments from the submit line of an sbatch script.

    Input:
     - submit_line (str): The submit line string being analyzed
     - long (str): The long version of the name (such as --nodelist)
     - short (str): The short version of the name (such as -w)

    Example input:
    submit_line: 'sbatch --reservation=example_res --account=anonaccount --mem=240G --partition=anonpartition --exclusive --hint=samplehint -N 1 -w x1000c0s0b0n1 -t 8:00:00 ./sample.sh'
    long: '--nodelist'
    short: '-w'

    Example output:
    target_arg: 'x1000c0s0b0n1'
    """
    if not submit_line:
        return None
    words = submit_line.strip(" ").split()
    target_arg = None
    
    for i_previous_word, word in enumerate(words[1:]):
        # Batch script or executable marks end of options
        if word[0] != "-" and (words[i_previous_word][0] != "-" or "=" in words[i_previous_word]):
            break
        if long:
            if long + "=" in word:
                target_arg = word.split(long + "=")[1]
                break
            if word == long:
                target_arg = words[i_previous_word + 2] 
                break
        if short:
            if word == short:
                target_arg = words[i_previous_word + 2]
                break

    return target_arg


def timelimit_str_to_timedelta(t_str):
    """
    Convert a Timelimit string to a Timedelta object.

    Example input: 2-00:00:00
    Example output: timedelta(2, 0, 0, 0)
    """
    days, hrs = 0, 0
    if "-" in t_str:
        days = int(t_str.split("-")[0])
        t_str = t_str.split("-")[1]

    if t_str.count(":") == 1 and t_str.count("."): # MM:SS.SS
        mins, secs = t_str.split(":")
        mins = int(mins)
        secs = float(secs)
    elif t_str.count(":") == 2: ## HH:MM:SS (SS has no decimal place for these ones)
        hrs, mins, secs = map(int, t_str.split(":"))

    return timedelta(days=days, hours=hrs, minutes=mins, seconds=secs)


def convert_to_raw(df, cols):
    """
    Convert a string with a metric prefix character (K, M, G, T) to an integer value.
    """
    df[cols] = df[cols].astype(str)
    df[cols] = df[cols].replace(
        { "K" : "e+03", "M" : "e+06", "G" : "e+09", "T" : "e+12" }, regex=True
    ).astype(float).astype(int)
    return df
    

def split_and_mask_events(events):
    """
    Split overlapping events, so that the most recently started event is considered.
    This is helpful when there are overlapping reservations on a node, which can occur
    in instances where a node goes down while in a reservation, and an administrator
    puts it in a 'repair' reservation while the issue is diagnosed.

    Expects events of the form (start, end, name, impromptu)

    impromptu is a boolean declaring this event as either impromptu (True) or known-in-advance (False)
    """
    # Sort events based on their start time
    events.sort(key=lambda x: x[0])

    # Initialize the list to store the final processed events
    processed_events = []

    for event in events:
        start, end, name, impromptu = event

        # Create a temporary list to hold the new event segments
        new_segments = []

        for prev_start, prev_end, prev_name, prev_impromptu in processed_events:
            if prev_start < end and start < prev_end:
                # Overlap case: split the previous event
                if start > prev_start:
                    new_segments.append((prev_start, start, prev_name, prev_impromptu))
                if end < prev_end:
                    new_segments.append((end, prev_end, prev_name, prev_impromptu))
            else:
                # No overlap, keep the previous event segment
                new_segments.append((prev_start, prev_end, prev_name, prev_impromptu))

        # Add the current event as its own segment
        new_segments.append((start, end, name, impromptu))

        # Update the processed events list with the new segments
        processed_events = new_segments

    processed_events = sorted(processed_events, key=lambda x: x[0], reverse=True)

    return processed_events


def mark_skip(job, now, reason):
    if job.last_skip == reason:                # avoid spamming
        return
    job.last_skip = reason
    job.wait_history.append((now, reason))