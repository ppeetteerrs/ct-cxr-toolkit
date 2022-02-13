from covid_ct.metadata.parse_dcm_fields import parse_dcm_fields
from covid_ct.metadata.select_dcms import select_dcms

if __name__ == "__main__":
    print("Parsing metadata...")
    parse_dcm_fields()

    print("Filtering metadata")
    select_dcms()
