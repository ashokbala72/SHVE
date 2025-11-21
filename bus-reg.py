import random
import pandas as pd
from faker import Faker

# Initialize Faker instance for generating random names
fake = Faker()

# List of Italian cities
italian_cities = [
    "Rome", "Milan", "Naples", "Turin", "Palermo", "Genoa", "Bologna", "Florence", "Venice", "Verona",
    "Messina", "Padua", "Trieste", "Bari", "Catania", "Brescia", "Reggio Calabria", "Modena", "Cagliari", "Parma"
]

# List of expertise in off-grid energy (you can expand it)
expertise_list = [
    "Solar Power", "Wind Energy", "Battery Storage", "Off-Grid Solutions", "Renewable Energy Solutions", "Energy Efficiency"
]

# Function to generate synthetic salesperson data
def generate_synthetic_sales_data(num_salespersons):
    sales_data = []
    
    for _ in range(num_salespersons):
        salesperson_id = f"SP-{random.randint(1000, 9999)}"  # Generate a unique Salesperson ID
        name = fake.name()  # Generate a random name
        experience = random.randint(1, 20)  # Random experience between 1 and 20 years
        expertise = random.choice(expertise_list)  # Randomly select expertise
        city = random.choice(italian_cities)  # Randomly select city in Italy
        
        sales_data.append({
            "Sales Person ID": salesperson_id,
            "Name": name,
            "Experience (Years)": experience,
            "Expertise in Off-Grid Energy": expertise,
            "Location (City in Italy)": city
        })
    
    return sales_data

# Generate data for 100 salespeople (you can change the number as needed)
num_salespersons = 100
sales_data = generate_synthetic_sales_data(num_salespersons)

# Create a DataFrame
sales_df = pd.DataFrame(sales_data)

# Save the DataFrame to a CSV file
csv_file_path = "synthetic_sales_data.csv"
sales_df.to_csv(csv_file_path, index=False)

# Print success message
print(f"Synthetic sales data CSV has been created and saved as '{csv_file_path}'!")
