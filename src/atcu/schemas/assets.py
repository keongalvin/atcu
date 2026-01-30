import enum


class Sector(enum.StrEnum):
    """TradingView sector classifications."""

    ELECTRONIC_TECHNOLOGY = enum.auto()  # "Electronic Technology"
    TECHNOLOGY_SERVICES = enum.auto()  # "Technology Services"


class Industry(enum.StrEnum):
    """TradingView industry classifications."""

    # Electronic Technology sector
    SEMICONDUCTORS = enum.auto()  # "Semiconductors"
    ELECTRONIC_PRODUCTION_EQUIPMENT = enum.auto()  # "Electronic Production Equipment"
    TELECOMMUNICATIONS_EQUIPMENT = enum.auto()  # "Telecommunications Equipment"
    COMPUTER_PROCESSING_HARDWARE = enum.auto()  # "Computer Processing Hardware"
    ELECTRONIC_COMPONENTS = enum.auto()  # "Electronic Components"

    # Technology Services sector
    PACKAGED_SOFTWARE = enum.auto()  # "Packaged Software"
    INTERNET_SOFTWARE_SERVICES = enum.auto()  # "Internet Software/Services"
    INFORMATION_TECHNOLOGY_SERVICES = enum.auto()  # "Information Technology Services"
    DATA_PROCESSING_SERVICES = enum.auto()  # "Data Processing Services"


class SubIndustry(enum.StrEnum):
    """Sub-industry classifications for finer granularity."""

    # Semiconductors sub-industries
    GPU_AI = enum.auto()  # "GPU/AI Accelerators"
    MEMORY = enum.auto()  # "Memory (DRAM/NAND/HBM)"
    ANALOG = enum.auto()  # "Analog Semiconductors"
    RF = enum.auto()  # "RF Semiconductors"
    FOUNDRY = enum.auto()  # "Foundry/Manufacturing"
    IP_DESIGN = enum.auto()  # "IP/Design Services"
    AUTOMOTIVE_IOT = enum.auto()  # "Automotive/IoT Chips"
    POWER_MANAGEMENT = enum.auto()  # "Power Management"
    FPGA = enum.auto()  # "FPGAs"
    MICROCONTROLLERS = enum.auto()  # "Microcontrollers"

    # Electronic Production Equipment sub-industries
    DEPOSITION_ETCH = enum.auto()  #  "Deposition/Etch"
    LITHOGRAPHY = enum.auto()  #  "Lithography"
    INSPECTION_METROLOGY = enum.auto()  #  "Inspection/Metrology"

    # Telecommunications Equipment sub-industries
    NETWORKING = enum.auto()  #  "Networking/Switching"
    OPTICAL = enum.auto()  #  "Optical Transport"

    # EDA Software (within Packaged Software)
    EDA_SOFTWARE = enum.auto()  # "EDA Software"


INDUSTRY_SECTOR_MAP: dict[Industry, Sector] = {
    Industry.SEMICONDUCTORS: Sector.ELECTRONIC_TECHNOLOGY,
    Industry.ELECTRONIC_PRODUCTION_EQUIPMENT: Sector.ELECTRONIC_TECHNOLOGY,
    Industry.TELECOMMUNICATIONS_EQUIPMENT: Sector.ELECTRONIC_TECHNOLOGY,
    Industry.COMPUTER_PROCESSING_HARDWARE: Sector.ELECTRONIC_TECHNOLOGY,
    Industry.ELECTRONIC_COMPONENTS: Sector.ELECTRONIC_TECHNOLOGY,
    Industry.PACKAGED_SOFTWARE: Sector.TECHNOLOGY_SERVICES,
    Industry.INTERNET_SOFTWARE_SERVICES: Sector.TECHNOLOGY_SERVICES,
    Industry.INFORMATION_TECHNOLOGY_SERVICES: Sector.TECHNOLOGY_SERVICES,
    Industry.DATA_PROCESSING_SERVICES: Sector.TECHNOLOGY_SERVICES,
}
