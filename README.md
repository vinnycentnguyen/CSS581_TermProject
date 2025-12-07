### Notes
This python package uses a predownloaded VIX CSV file and pulls data from the Deribit API to conduct ML regression testing.

You will need the VIX_History.csv as a pre-requisite.

To run the program:

1. `pip install pandas`
2. `pip install scikit-learn`
3. `pip install matplotlib`
4. `pip install numpy`
5. `python main.py`

This will download the necessary libraries and run the main program, which will fetch the Bitcoin data, conduct preprocessing, and execute models based on the processed data.