from utils.config import CONFIG

from covid_ct.dataset.drr import covid_ct_drr

if __name__ == "__main__":

    # print("Generating train dataset...")
    # covid_ct_drr(CONFIG.OUTPUT_DIR / "covid_ct/train")

    print("Generating test dataset...")
    covid_ct_drr(CONFIG.OUTPUT_DIR / "covid_ct/test")
