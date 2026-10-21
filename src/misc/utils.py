def get_overlap_tag(overlap):
    if 0.05 <= overlap <= 0.3:
        overlap_tag = "small"
    elif overlap <= 0.55:
        overlap_tag = "medium"
    elif overlap <= 0.8:
        overlap_tag = "large"
    else:
        overlap_tag = "ignore"

    return overlap_tag
