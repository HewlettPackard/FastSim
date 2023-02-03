from datetime import timedelta


def convert_nodelist_to_node_nums(nid_str):
    node_nums = []

    nid_str = nid_str.strip("nid").strip("[").strip("]")
    for nid_str_entry in nid_str.split(","):
        if "-" not in nid_str_entry:
            node_nums.append(int(nid_str_entry))
            continue

        nid_str_range = nid_str_entry.split("-")
        for node_num in range(int(nid_str_range[0]), int(nid_str_range[1]) + 1):
            node_nums.append(node_num)

    return node_nums


def get_sbatch_cli_arg(submit_line, long="", short=""):
    words = submit_line.split(" ")
    dep_arg = None
    for i_last_word, word in enumerate(words[1:]):
        # Batch script or executable marks end of options
        if word[0] != "-" and (words[i_last_word][0] != "-" or "=" in words[i_last_word]):
            break
        if long:
            if long + "=" in word:
                dep_arg = word.split(long + "=")[1]
                break
            if word == long:
                dep_arg = words[i_last_word + 2]
                break
        if short:
            if word == short:
                dep_arg = words[i_last_word + 2]
                break

    return dep_arg


def timelimit_str_to_timedelta(t_str):
    days, hrs = 0, 0
    try:
        if "-" in t_str:
            days = int(t_str.split("-")[0])
            t_str = t_str.split("-")[1]
    except:
        print(t_str)

    if t_str.count(":") == 1 and t_str.count("."): # MM:SS.SS
        mins, secs = t_str.split(":")
        mins = int(mins)
        secs = float(secs)
    elif t_str.count(":") == 2: ## HH:MM:SS (SS has no decimal place for these ones)
        hrs, mins, secs = map(int, t_str.split(":"))
    else:
        raise NotImplementedError("Bruh")

    return timedelta(days=days, hours=hrs, minutes=mins, seconds=secs)


def convert_to_raw(df, cols):
    df[cols] = df[cols].astype(str)
    df[cols] = df[cols].replace(
        { "K" : "e+03", "M" : "e+06", "G" : "e+09", "T" : "e+12" }, regex=True
    ).astype(float).astype(int)
    return df

