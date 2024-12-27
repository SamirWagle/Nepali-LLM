# for better viewing and later on training purposes 
import pandas as pd


def csv_to_xlsx(csv_path, xlsx_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Write to an Excel file
    df.to_excel(xlsx_path, index=False)

if __name__ == "__main__":
    # Example usage
    csv_path = "All Scraped Data.csv"
    xlsx_path = "All Scraped Data.xlsx"
    
    # Ensure the directory for the output file exists
  
    
    csv_to_xlsx(csv_path, xlsx_path)
    