def truncate_to_minute(dt):
    return dt.replace(second=0, microsecond=0)
