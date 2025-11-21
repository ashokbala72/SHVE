import requests
import streamlit as st
import os
import json
import pandas as pd
from dotenv import load_dotenv
import re
import random

# Load environment variables
load_dotenv()

# Azure OpenAI configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# Path to the leads CSV
leads_csv_path = "leads.csv"
customers_csv_path = "customers.csv"

# Load data from CSV (Restaurants data)
def load_data():
    restaurants_csv_path = "restaurants_italy.csv"
    if os.path.exists(restaurants_csv_path):
        restaurants_df = pd.read_csv(restaurants_csv_path)
    else:
        restaurants_df = pd.DataFrame(columns=["Name", "Address", "Type", "Popularity", "Profit"])
    return restaurants_df

# Load leads data (if it exists)
def load_leads_data():
    if os.path.exists(leads_csv_path):
        return pd.read_csv(leads_csv_path)
    else:
        return pd.DataFrame(columns=["Rank", "Name", "Address", "Profit", "Popularity", "Market Share", "Credit Score", "Location Rating", "Select"])

# Load salesperson data (from synthetic_sales_data.csv)
def load_salesperson_data():
    if os.path.exists("synthetic_sales_data.csv"):
        return pd.read_csv("synthetic_sales_data.csv")
    else:
        st.error("Salesperson data (synthetic_sales_data.csv) is missing!")
        return pd.DataFrame(columns=["Sales Person ID", "Name", "Experience (Years)", "Expertise in Off-Grid Energy", "Location (City in Italy)"])

import openai
import requests

# Function to get the best salesperson from OpenAI based on the business and expertise
# Function to get the best salesperson from OpenAI based on the business and expertise
def get_salesperson_recommendation(business_name, expertise_needed, leads_df, sales_df):
    # Construct the prompt for OpenAI
    prompt = f"""
    We have a business lead named '{business_name}' that requires expertise in {expertise_needed}. Please recommend the most suitable salesperson from the following list based on their expertise, experience, and location. 
    The response should be in valid JSON format with keys:
    - "Sales Person ID"
    - "Name"
    - "Experience"
    - "Expertise"
    - "Location"
    
    LEAD:
    Business Name: {business_name}
    Expertise Needed: {expertise_needed}
    
    SALESPEOPLE:
    """

    # Add the salespeople data to the prompt
    for _, row in sales_df.iterrows():
        prompt += f"\nSales Person ID: {row['Sales Person ID']}, Name: {row['Name']}, Experience: {row['Experience (Years)']} years, Expertise: {row['Expertise in Off-Grid Energy']}, Location: {row['Location (City in Italy)']}"

    prompt += """
    PLEASE RESPOND IN THE EXACT JSON FORMAT BELOW:
    {
        "Sales Person ID": "[ID]",
        "Name": "[Name]",
        "Experience": "[Experience (Years)]",
        "Expertise": "[Expertise]",
        "Location": "[Location]"
    }
    ONLY provide the data in the JSON format above. Do not include any extra explanation, text, or additional information.
    """

    # Send the prompt to OpenAI for recommendation
    headers = {
        "Authorization": f"Bearer {AZURE_OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0.5,
    }

    try:
        # Call OpenAI API to get the recommendation
        response = requests.post(
            f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}",
            headers=headers,
            json=body,
            timeout=30,
        )

        if response.status_code == 200:
            # Get the response from OpenAI (the content is inside this structure)
            response_data = response.json()  # This is a dictionary
            recommended_salesperson = response_data["choices"][0]["message"]["content"].strip()

            try:
                # Try to parse the response as JSON
                salesperson_data = json.loads(recommended_salesperson)

                # Ensure the JSON response contains the expected fields
                if all(field in salesperson_data for field in ["Sales Person ID", "Name", "Experience", "Expertise", "Location"]):
                    return salesperson_data
                else:
                    st.warning(f"Missing expected fields in the response for business '{business_name}'.")
                    return None
            except json.JSONDecodeError:
                st.warning(f"Error decoding OpenAI response for business '{business_name}'. Response is not valid JSON.")
                return None
        else:
            st.error(f"Error with OpenAI API: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error with OpenAI request: {e}")
        return None
# Function to generate synthetic data batch using OpenAI
def generate_synthetic_data_batch(business_names, country="IT"):
    prompt = """
    Please provide the following data for each restaurant in valid JSON format:
    - Use proper commas between items.
    - Ensure no trailing commas at the end of the array or objects.
    - All keys and string values should be properly quoted (double quotes).
    - Every JSON object must be properly structured and have the necessary commas to separate each entry in the array.
    - The response should be a JSON array with each object representing a restaurant and having the following fields: "business_name", "estimated_revenue", "market_share", "credit_score", "location_rating".
    
    For the "estimated_revenue" and "market_share", please ensure the following:
    - **"estimated_revenue"** should be a number without any symbols like "€". Just provide the number, e.g., "1200000" (without the currency symbol).
    - **"market_share"** should be a plain numeric value without the "%" symbol. For example, "2.5" instead of "2.5%".
    - **"credit_score"** should be a plain number between 0 and 100, e.g., "87" instead of "780".
    - **"location_rating"** should be a number between 0 and 5.
    - Ensure the JSON array is properly closed with `]` at the end and there are no extra characters like `}`.

    The format should look like this:
    [
        {
            "business_name": "Business Name",
            "estimated_revenue": 1200000,
            "market_share": 2.5,
            "credit_score": 86,
            "location_rating": 4.5
        },
        {
            "business_name": "Another Business",
            "estimated_revenue": 1500000,
            "market_share": 3.0,
            "credit_score": 90,
            "location_rating": 4.8
        }
    ]
    
    Do not include any additional text or formatting other than proper JSON. Ensure the JSON array is correctly closed with `]`.
    """

    for name in business_names:
        prompt += f"Business Name: {name}\n"

    headers = {
        "Authorization": f"Bearer {AZURE_OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 5000,
        "temperature": 0.5,
    }

    try:
        with st.spinner(f"Generating synthetic data for the batch of businesses..."):
            response = requests.post(
                f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}",
                headers=headers,
                json=body,
                timeout=30,
            )

            if response.status_code != 200:
                st.error(f"OpenAI API Error: {response.status_code} - {response.text}")
                return []

            try:
                msg = response.json()["choices"][0]["message"]["content"].strip()

                # Clean up the response
                msg_cleaned = re.sub(r"^```json|```$", "", msg).strip()
                if msg_cleaned.endswith("},"):
                    msg_cleaned = msg_cleaned[:-1]  # Remove trailing comma after the last object
                if msg_cleaned.endswith("] }"):
                    msg_cleaned = msg_cleaned[:-2] + "]"  # Remove extra closing brace after the last object
                if not msg_cleaned.endswith("]"):
                    msg_cleaned = msg_cleaned.rstrip(',') + "]"  # Ensure proper closing of JSON array

                data = json.loads(msg_cleaned)  # Parse JSON response
                return data

            except json.JSONDecodeError as e:
                st.error(f"Error decoding OpenAI response as JSON: {e}")
                return []

    except Exception as e:
        st.error(f"Error during OpenAI API request: {e}")
        return []

# Function to get the rank from OpenAI based on synthetic data
def get_rank_from_openai(synthetic_data):
    """
    Use OpenAI to calculate the rank for a business based on its synthetic data.
    """
    if not synthetic_data:
        return random.randint(1, 100)  # Fallback if no data is available

    # Prepare prompt to explicitly request a rank between 1 and 100
    prompt = f"""
    Based on the following synthetic data for the business, please provide a rank between 1 and 100, where 1 is the best and 100 is the worst:
    - Estimated Revenue: {synthetic_data.get("estimated_revenue")}
    - Market Share: {synthetic_data.get("market_share")}
    - Credit Score: {synthetic_data.get("credit_score")}
    - Location Rating: {synthetic_data.get("location_rating")}

    Please return only the rank number, without any additional explanation or text. 
    """

    headers = {
        "Authorization": f"Bearer {AZURE_OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 50,
        "temperature": 0.5,
    }

    try:
        with st.spinner(f"Getting rank from OpenAI..."):
            response = requests.post(
                f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}",
                headers=headers,
                json=body,
                timeout=30,
            )

            if response.status_code != 200:
                st.error(f"OpenAI API Error: {response.status_code} - {response.text}")
                return None

            try:
                rank_response = response.json()["choices"][0]["message"]["content"].strip()
                rank = int(rank_response)  # Attempt to convert the response directly to an integer
                return min(max(rank, 1), 100)  # Ensure the rank is between 1 and 100

            except json.JSONDecodeError as e:
                st.error(f"Error decoding OpenAI rank response: {e}")
                return None
            except ValueError as e:
                st.error(f"Error: Invalid rank format returned by OpenAI: {e}")
                return None

    except Exception as e:
        st.error(f"Error during OpenAI API request: {e}")
        return None


# Streamlit Layout (remains the same as your previous code)
st.set_page_config(page_title="SHV Energy Lead Management", page_icon="🔥", layout="wide")

# Custom styling for title and subtitle, adding table styles to ensure alternating row colors
st.markdown("""
    <style>
        .stTitle h1 {
            font-size: 30px;
            text-align: center;
        }
        .stSubtitle h2 {
            font-size: 18px;
            text-align: center;
        }
        .table-container {
            overflow-x: auto;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:nth-child(even) {
            background-color: #e0f7fa;  /* Light blue for even rows */
        }
        tr:nth-child(odd) {
            background-color: #e8f5e9;  /* Light green for odd rows */
        }
    </style>
""", unsafe_allow_html=True)

# Title and Subtitle with custom styles
st.markdown('<div class="stTitle"><h1>SHV Energy Prospect & Lead Agent</h1></div>', unsafe_allow_html=True)
st.markdown('<div class="stSubtitle"><h2>Powered by TCS GEN AI and Azure Open AI</h2></div>', unsafe_allow_html=True)

# Sidebar for tab selection
st.sidebar.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
tab_selection = st.sidebar.radio("Select a Tab", ["Prospects", "Leads", "Assignment", "Lead Information", "Sales Email"])

# Process prospects and add checkboxes
# Process prospects and add checkboxes
# Custom styling for table, adding borders and alternating row colors (light green and light blue)
st.markdown("""
    <style>
        .table-container {
            overflow-x: auto;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            border: 1px solid #ddd;  /* Border for the whole table */
        }
        th, td {
            padding: 8px;
            text-align: left;
            border: 1px solid #ddd;  /* Border for each cell */
        }
        th {
            background-color: #f2f2f2;
        }
        tr:nth-child(even) {
            background-color: #e0f7fa;  /* Light blue for even rows */
        }
        tr:nth-child(odd) {
            background-color: #e8f5e9;  /* Light green for odd rows */
        }
    </style>
""", unsafe_allow_html=True)

if tab_selection == "Prospects":
    st.title("Prospects")

    # Load restaurant data
    restaurants_df = load_data()

    # Load customer and leads data to exclude already existing businesses
    customers_df = load_leads_data()  # Use the same function for loading customers and leads
    leads_df = load_leads_data()

    # Combine customers and leads data to get a list of already existing businesses
    existing_businesses = pd.concat([customers_df, leads_df])['Name'].tolist()

    # Filter out the businesses that are already in customers or leads CSVs
    filtered_restaurants_df = restaurants_df[~restaurants_df['Name'].isin(existing_businesses)]

    # Check if synthetic data is already in session_state
    if 'synthetic_data_batch' not in st.session_state:
        # Generate synthetic data for all businesses and store it in session_state
        business_names_batch = filtered_restaurants_df['Name'].tolist()
        synthetic_data_batch = generate_synthetic_data_batch(business_names_batch)
        st.session_state.synthetic_data_batch = synthetic_data_batch  # Store it in session_state

    # Use the synthetic data from session_state
    synthetic_data_batch = st.session_state.synthetic_data_batch
    prospects_selection = []

    if synthetic_data_batch:
        for idx, row in filtered_restaurants_df.iterrows():
            # Get synthetic data for the corresponding business
            synthetic_data = next((item for item in synthetic_data_batch if item["business_name"] == row["Name"]), None)

            if synthetic_data:
                # Get the rank from OpenAI
                rank = get_rank_from_openai(synthetic_data)
                if rank is not None:
                    prospects_selection.append({
                        "Rank": rank,
                        "Name": row["Name"],
                        "Address": row["Address"],
                        "Profit": synthetic_data.get("estimated_revenue", "Not Available"),
                        "Popularity": row["Popularity"] if pd.notna(row["Popularity"]) else "Not Available",
                        "Market Share": synthetic_data.get("market_share", "Not Available"),
                        "Credit Score": synthetic_data.get("credit_score", "Not Available"),
                        "Location Rating": synthetic_data.get("location_rating", "Not Available")
                    })

        # Convert the list of prospects to a DataFrame
        prospects_df = pd.DataFrame(prospects_selection)

        # Sort the prospects by rank (lower rank means better)
        prospects_df = prospects_df.sort_values(by=["Rank"], ascending=True)

        # Manually display the headers before the rows
        cols = st.columns([1, 3, 3, 2, 2, 2, 2, 2, 1])
        cols[0].write("Rank")
        cols[1].write("Name")
        cols[2].write("Address")
        cols[3].write("Profit")
        cols[4].write("Popularity")
        cols[5].write("Market Share")
        cols[6].write("Credit Score")
        cols[7].write("Location Rating")
        cols[8].write("Select")

        # Add checkboxes and handle selected prospects
        prospects_df['Select'] = False  # Add a new column to store checkbox state
        for idx, row in prospects_df.iterrows():
            cols = st.columns([1, 3, 3, 2, 2, 2, 2, 2, 1])
            cols[0].write(row['Rank'])
            cols[1].write(row['Name'])
            cols[2].write(row['Address'])
            cols[3].write(row['Profit'])
            cols[4].write(row['Popularity'])
            cols[5].write(row['Market Share'])
            cols[6].write(row['Credit Score'])
            cols[7].write(row['Location Rating'])
            
            checkbox = cols[8].checkbox("", key=f"checkbox_{idx}")  # Create checkbox for each row
            # Store the checkbox state in the DataFrame
            prospects_df.at[idx, 'Select'] = checkbox

        # Generate leads when the button is clicked
        selected_prospects = prospects_df[prospects_df["Select"]]

        if st.button("Generate Selected Leads"):
            if len(selected_prospects) > 0:
                # Load the original restaurant data from restaurants_italy.csv
                restaurants_df = load_data()  # This reads from the restaurants_italy.csv

                # Merge the selected prospects with the original restaurants dataframe to match the row data
                selected_prospects_data = restaurants_df[restaurants_df['Name'].isin(selected_prospects['Name'])]

                # Append the selected prospects to leads.csv
                leads_df = load_leads_data()  # This loads the current leads data (if any)
                new_leads = pd.concat([leads_df, selected_prospects_data], ignore_index=True)

                # Save the updated leads data back to leads.csv
                new_leads.to_csv(leads_csv_path, index=False)

                st.success(f"Successfully added {len(selected_prospects)} selected prospects as Leads.")
            else:
                st.warning("Please select at least one prospect.")
# "Leads" tab logic
if tab_selection == "Leads":
    st.title("Leads")

    # Load leads data
    leads_df = load_leads_data()

    if not leads_df.empty:
        # Get synthetic data batch for leads from OpenAI
        business_names_batch = leads_df['Name'].tolist()
        synthetic_data_batch = generate_synthetic_data_batch(business_names_batch)

        leads_selection = []

        if synthetic_data_batch:
            for idx, row in leads_df.iterrows():
                # Get synthetic data for the corresponding business
                synthetic_data = next((item for item in synthetic_data_batch if item["business_name"] == row["Name"]), None)

                if synthetic_data:
                    # Get the rank from OpenAI
                    rank = get_rank_from_openai(synthetic_data)
                    if rank is not None:
                        leads_selection.append({
                            "Rank": rank,
                            "Name": row["Name"],
                            "Address": row["Address"],
                            "Profit": synthetic_data.get("estimated_revenue", "Not Available"),
                            "Popularity": row["Popularity"] if pd.notna(row["Popularity"]) else "Not Available",
                            "Market Share": synthetic_data.get("market_share", "Not Available"),
                            "Credit Score": synthetic_data.get("credit_score", "Not Available"),
                            "Location Rating": synthetic_data.get("location_rating", "Not Available")
                        })

        # Convert the list of leads to a DataFrame
        leads_df = pd.DataFrame(leads_selection)

        # Sort the leads by rank (lower rank means better)
        leads_df = leads_df.sort_values(by=["Rank"], ascending=True)

        # Manually display the headers before the rows
        cols = st.columns([1, 3, 3, 2, 2, 2, 2, 2, 1])
        cols[0].write("Rank")
        cols[1].write("Name")
        cols[2].write("Address")
        cols[3].write("Profit")
        cols[4].write("Popularity")
        cols[5].write("Market Share")
        cols[6].write("Credit Score")
        cols[7].write("Location Rating")

        # Display the rows
        for idx, row in leads_df.iterrows():
            cols = st.columns([1, 3, 3, 2, 2, 2, 2, 2, 1])
            cols[0].write(row['Rank'])
            cols[1].write(row['Name'])
            cols[2].write(row['Address'])
            cols[3].write(row['Profit'])
            cols[4].write(row['Popularity'])
            cols[5].write(row['Market Share'])
            cols[6].write(row['Credit Score'])
            cols[7].write(row['Location Rating'])


if tab_selection == "Assignment":
    st.title("Assignment")

    # Load leads data and salesperson data
    leads_df = load_leads_data()
    sales_df = load_salesperson_data()

    # Prepare the data to display (business name, and recommendation for sales person)
    assignment_data = []

    # Process if there are leads and salesperson data available
    if not leads_df.empty and not sales_df.empty:
        # Loop over each lead
        for idx, lead in leads_df.iterrows():
            business_name = lead['Name']
            expertise_needed = "Off-Grid Solutions"  # Expertise fixed for now (can be dynamic if needed)

            # Generate prompt for OpenAI to get the most suitable salesperson
            prompt = f"""
            We have a business lead named '{business_name}' that requires expertise in {expertise_needed}. Please recommend the most suitable salesperson from the following list based on their expertise, experience, and location:
            
            LEAD:
            Business Name: {business_name}
            Expertise Needed: {expertise_needed}
            
            SALESPEOPLE:
            """

            # Add salespeople data to the prompt
            for _, row in sales_df.iterrows():
                prompt += f"\nSales Person ID: {row['Sales Person ID']}, Name: {row['Name']}, Experience: {row['Experience (Years)']} years, Expertise: {row['Expertise in Off-Grid Energy']}, Location: {row['Location (City in Italy)']}"

            prompt += "\n\nPlease recommend the best salesperson for this lead, providing the Salesperson ID, Name, Experience, Expertise, and Location."

            # Send the prompt to OpenAI for recommendation
            recommended_salesperson = get_salesperson_recommendation(business_name, expertise_needed, leads_df, sales_df)

            # Assuming recommended_salesperson is a dictionary (as the OpenAI response should be in JSON format)
            if recommended_salesperson:
                # If the response is in the correct format, it should already be a dictionary.
                # No need to split, simply access the dictionary fields directly:
                
                # Example of accessing values in the dictionary
                salesperson_id = recommended_salesperson.get("Sales Person ID", "N/A")
                salesperson_name = recommended_salesperson.get("Name", "N/A")
                experience = recommended_salesperson.get("Experience", "N/A")
                expertise = recommended_salesperson.get("Expertise", "N/A")
                location = recommended_salesperson.get("Location", "N/A")

                # Append the assignment data to the list
                assignment_data.append({
                    "Business Name": business_name,
                    "Location": lead['Address'],  # Location of the business (Address)
                    "Sales Person ID": salesperson_id,
                    "Sales Person Name": salesperson_name,
                    "Sales Person Location": location,
                    "Expertise": expertise,
                    "Experience": experience
                })
            else:
                st.warning(f"No salesperson recommendation received for business '{business_name}'.")
                
        # Display the assignment table with the recommended salesperson
        if assignment_data:
            assignment_df = pd.DataFrame(assignment_data)
            st.dataframe(assignment_df)
        else:
            st.warning("No assignments available.")

if tab_selection == "Lead Information":
    st.title("Lead Information")

    # Load leads data
    leads_df = load_leads_data()

    if leads_df.empty:
        st.warning("No leads found. Please generate leads first.")
    else:
        # Dropdown to select a business lead
        selected_lead = st.selectbox(
            "Select a Lead to View Details",
            leads_df["Name"].tolist(),
            index=0  # Default to the first lead
        )

        # Get selected business details
        business_info = leads_df[leads_df["Name"] == selected_lead].iloc[0]
        business_address = business_info.get("Address", "Not Available")

        # Display selected lead info summary
        st.markdown(f"### 🏢 {selected_lead}")
        st.markdown(f"**📍 Address:** {business_address}")
        st.markdown("---")

        # Function to get additional business info from OpenAI
        def get_business_information(business_name, business_address):
            prompt = f"""
            You are a business intelligence assistant. Provide detailed information about the company named "{business_name}" located at "{business_address}".
            Include the following if available:
            - Overview of the business
            - Industry and sector
            - Services or products offered
            - Company size or popularity
            - Estimated financial performance (approximate revenue/profit)
            - Recent news, reviews, or reputation
            - Competitive landscape and local market context

            Format the response as a clean, structured summary using Markdown headings and bullet points.
            """

            headers = {
                "Authorization": f"Bearer {AZURE_OPENAI_API_KEY}",
                "Content-Type": "application/json",
            }
            body = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 800,
                "temperature": 0.6,
            }

            try:
                response = requests.post(
                    f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}",
                    headers=headers,
                    json=body,
                    timeout=40,
                )

                if response.status_code == 200:
                    data = response.json()
                    ai_output = data["choices"][0]["message"]["content"].strip()
                    return ai_output
                else:
                    st.error(f"OpenAI API Error: {response.status_code}")
                    return None
            except Exception as e:
                st.error(f"Error fetching business information: {e}")
                return None

        # Fetch data from OpenAI
        with st.spinner(f"Fetching detailed information about {selected_lead}..."):
            business_details = get_business_information(selected_lead, business_address)

        # Display the formatted response
        if business_details:
            st.markdown("### 🧠 Business Intelligence Summary")
            st.markdown(business_details)
        else:
            st.warning("No additional information could be retrieved for this business.")

import streamlit as st
import requests
import json
import pyperclip  # Optional: for local copy functionality if desired

if tab_selection == "Sales Email":
    st.title("Sales Email Generator")

    # Load data
    leads_df = load_leads_data()
    sales_df = load_salesperson_data()

    if leads_df.empty:
        st.warning("No leads available. Please generate leads first.")
    else:
        # Dropdown for selecting lead
        selected_lead = st.selectbox(
            "Select a Lead",
            leads_df["Name"].tolist(),
            index=0
        )

        # Fetch the chosen lead’s basic details
        lead_info = leads_df[leads_df["Name"] == selected_lead].iloc[0]
        business_address = lead_info.get("Address", "Not Available")

        # Fetch assigned salesperson from Assignment tab logic
        # You can load it from CSV if persisted; here we pick one randomly for demo
        assigned_salesperson = sales_df.sample(1).iloc[0]
        salesperson_name = assigned_salesperson["Name"]
        salesperson_experience = assigned_salesperson["Experience (Years)"]
        salesperson_expertise = assigned_salesperson["Expertise in Off-Grid Energy"]
        salesperson_location = assigned_salesperson["Location (City in Italy)"]

        # Get contextual business info for personalization
        def get_business_summary(business_name, business_address):
            prompt = f"""
            Provide a short summary (2–3 sentences) about the business '{business_name}', located at '{business_address}'.
            Focus on:
            - What the business is known for
            - Any reputation, achievements, or recent positive news
            Return only natural descriptive sentences suitable for including in a sales email.
            """

            headers = {
                "Authorization": f"Bearer {AZURE_OPENAI_API_KEY}",
                "Content-Type": "application/json",
            }
            body = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 250,
                "temperature": 0.7,
            }

            try:
                response = requests.post(
                    f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}",
                    headers=headers,
                    json=body,
                    timeout=30,
                )
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"].strip()
                else:
                    st.error(f"OpenAI API Error: {response.status_code}")
                    return ""
            except Exception as e:
                st.error(f"Error fetching business context: {e}")
                return ""

        # Get AI-generated business context
        with st.spinner(f"Gathering background on {selected_lead}..."):
            business_context = get_business_summary(selected_lead, business_address)

        # Generate a personalized sales email
        def generate_sales_email(business_name, business_context, salesperson_name, experience, expertise, location):
            prompt = f"""
            Write a professional, friendly outreach email from SHV Energy addressed to the owners or managers of '{business_name}'.
            - The sender is {salesperson_name} from SHV Energy, based in {location}.
            - Mention their {experience} years of experience in {expertise}.
            - Incorporate this business background naturally: {business_context}
            - Invite the business for a meeting to discuss SHV Energy’s Off-Grid Solutions and how they could help.
            - Format with multiple well-written paragraphs, professional tone, and clear paragraph spacing.
            - Close with a polite sign-off and the salesperson’s signature.
            """

            headers = {
                "Authorization": f"Bearer {AZURE_OPENAI_API_KEY}",
                "Content-Type": "application/json",
            }
            body = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 600,
                "temperature": 0.7,
            }

            try:
                response = requests.post(
                    f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}",
                    headers=headers,
                    json=body,
                    timeout=40,
                )

                if response.status_code == 200:
                    data = response.json()
                    email_text = data["choices"][0]["message"]["content"].strip()
                    return email_text
                else:
                    st.error(f"OpenAI API Error: {response.status_code}")
                    return None
            except Exception as e:
                st.error(f"Error generating email: {e}")
                return None

        # Fetch generated email
        with st.spinner(f"Generating personalized email for {selected_lead}..."):
            email_output = generate_sales_email(
                selected_lead, business_context,
                salesperson_name, salesperson_experience,
                salesperson_expertise, salesperson_location
            )

        # Display the generated email
        if email_output:
            st.markdown("### ✉️ Generated Sales Email")
            st.markdown(email_output)

            # Copy-to-clipboard button
            if st.button("📋 Copy Email to Clipboard"):
                pyperclip.copy(email_output)
                st.success("Email copied to clipboard!")
        else:
            st.warning("No email could be generated.")




