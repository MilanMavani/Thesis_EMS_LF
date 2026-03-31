# src/column_metadata.py

COLUMN_METADATA = {
    "BU_TotActPwr_Academy": {"unit": "kW"},
    "BU_TotActPwr_Tech_Room": {"unit": "kW"},
    "BU_TotActPwr_Pump_Room": {"unit": "kW"},
    "BU_TotActPwr_SDB_EL_Substation": {"unit": "kW"},
    "BA_TotActPwr_BESS_AC_Panel1": {"unit": "kW"},
    "BA_TotActPwr_BESS_AC_Panel2": {"unit": "kW"},
    "BU_TotActPwr_UPS2": {"unit": "kW"},
    "BU_TotActPwr_UPS1": {"unit": "kW"},
    "BU_TotActPwr_ELY_Supply": {"unit": "kW"},
    "BU_TotActPwr_Comp1": {"unit": "kW"},
    "BU_TotActPwr_Comp2": {"unit": "kW"},
    "BU_TotActPwr_Comp3": {"unit": "kW"},
    "BU_TotActPwr_NitrogenUnit": {"unit": "kW"},
    "BU_TotActPwr_Ely_BoP": {"unit": "kW"},
    "BU_TotActPwr_Chiller": {"unit": "kW"},
    "AuxPowCons": {"unit": "kW"},
    "BU_TotPwrReq": {"unit": "kW"},
    "BU_Unitstate": {"unit": ""},
    "BatSocAvg": {"unit": "%"},
    "BatPwrAtTot": {"unit": "kW"},
    "EgyDisMaxTot": {"unit": "kWh"},
    "EgyChrMaxTot": {"unit": "kWh"},
    "DevStt": {"unit": ""},
    "WS_AirTemp": {"unit": "°C"},
    "WS_Radiation": {"unit": "W/m²"},
    "WS_RelHum": {"unit": "%RH"},
    "WS_RelAirPre": {"unit": "hPa"},
    "PV_Unitstate": {"unit": ""},
    "BA_Unitstate": {"unit": ""},
}

def get_unit(col_name: str) -> str:
    return COLUMN_METADATA.get(col_name, {}).get("unit", "")

def get_plot_label(col_name: str) -> str:
    unit = get_unit(col_name)
    return f"{col_name} [{unit}]" if unit else col_name