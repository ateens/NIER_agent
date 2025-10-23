MATTER_DNSTY_MAP = {
    "CO": "co_dnsty",
    "O3": "oz_dnsty",
    "SO2": "so2_dnsty",
    "NOX": "nox_dnsty",
    "NO2": "no2_dnsty",
    "NO": "nmo_dnsty",
    "PM10": "pm_dnsty",
    "PM25": "pm25_dnsty",
}

MATTER_FLAG_MAP = {
    "CO": "co_dtl_flag",
    "O3": "oz_dtl_flag",
    "SO2": "so2_dtl_flag",
    "NOX": "nox_dtl_flag",
    "NO2": "no2_dtl_flag",
    "NO": "nmo_dtl_flag",
    "PM10": "pm_dtl_flag",
    "PM25": "pm25_dtl_flag",
}

CNTMN_CODE_MAP = {
    1: "SO2",
    2: "NO2",
    3: "O3",
    4: "CO",
    5: "PM10",
    6: "PM25",
    7: "NO",
    8: "NOX",
}


def dnsty_of(element, item):
    attr = MATTER_DNSTY_MAP.get(element)
    if attr:
        return getattr(item, attr)
    raise ValueError(f"Unsupported element type: {element}")
