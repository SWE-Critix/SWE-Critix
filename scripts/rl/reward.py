def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    solution_str = solution_str.strip()
    if solution_str.endswith("</think>\n\n<judgment>YES</judgment>") and ground_truth == '1':
        return 1.0

    if solution_str.endswith("</think>\n\n<judgment>NO</judgment>") and ground_truth == '0':
        return 1.0

    return -1.0
