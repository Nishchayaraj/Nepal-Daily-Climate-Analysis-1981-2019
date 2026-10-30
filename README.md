\# Climate zones and forecasting in Nepal (MINspire 2026 Capstone)



Team: \CAT-MAN-DU



Data: Nepal Daily Climate 1981-2019 (Kaggle, saimondahal). Download it, extract

dailyclimate.csv into this folder.



Run order (each script takes "." as the folder):

python prepare\_data.py dailyclimate.csv   (creates the three CSV files; copy them here)

python eda\_plots.py .

python clustering.py .

python forecasting.py .

python break\_analysis.py .

python zone\_forecasting.py .

python improved\_models.py .

python clustering\_robustness.py .

python zone\_robustness\_check.py .



Install libraries: pip install -r requirements.txt

