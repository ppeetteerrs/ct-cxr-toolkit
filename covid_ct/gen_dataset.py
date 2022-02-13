from utils.config import CONFIG

from covid_ct.dataset.run import covid_ct_dataset

if __name__ == "__main__":

    print("Generating train dataset...")
    covid_ct_dataset(CONFIG.OUTPUT_DIR / "covid_ct/train")

    print("Generating test dataset...")
    covid_ct_dataset(CONFIG.OUTPUT_DIR / "covid_ct/test")
