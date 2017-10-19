def get_len_dfs(list_dfs):
    return [len(p.index) for p in list_dfs]


def check_len_is_same(dfs_len, param):
    for len in dfs_len:
        assert param == len
