import requests
import streamlit as st
import os
import json
import pandas as pd
from dotenv import load_dotenv
import re
import random
import pyperclip  # Optional: for local copy functionality if desired

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
tab_selection = st.sidebar.radio("Select a Tab", [
    "Prospects", "Leads", "Assignment", "Lead Information", "Sales Email", 
    "Carbon Intensity Data", "Additional Insights", "Targeted Marketing Strategy"
])

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

    # Load customer and leads data to exclude businesses already added
    customers_df = load_leads_data()
    leads_df = load_leads_data()
    existing_businesses = pd.concat([customers_df, leads_df])['Name'].tolist()

    # Filter restaurants
    filtered_restaurants_df = restaurants_df[
        ~restaurants_df['Name'].isin(existing_businesses)
    ]

    # =========================================================
    # 1. Generate synthetic data ONCE and store in session state
    # =========================================================
    if "synthetic_data_batch" not in st.session_state:
        business_names = filtered_restaurants_df["Name"].tolist()
        st.session_state.synthetic_data_batch = generate_synthetic_data_batch(business_names)

    synthetic_data_batch = st.session_state.synthetic_data_batch

    # =========================================================
    # 2. Generate ranks ONCE per business and store in session state
    # =========================================================
    if "prospect_ranks" not in st.session_state:
        st.session_state.prospect_ranks = {}

        with st.spinner("Generating ranks for prospects..."):
            for item in synthetic_data_batch:
                business_name = item["business_name"]
                st.session_state.prospect_ranks[business_name] = get_rank_from_openai(item)

    ranks = st.session_state.prospect_ranks

    # =========================================================
    # 3. Build prospects table
    # =========================================================
    prospects_list = []
    for idx, row in filtered_restaurants_df.iterrows():
        syn = next((x for x in synthetic_data_batch if x["business_name"] == row["Name"]), None)
        if syn:
            prospects_list.append({
                "Rank": ranks.get(row["Name"], 999),
                "Name": row["Name"],
                "Address": row["Address"],
                "Profit": syn.get("estimated_revenue", "N/A"),
                "Popularity": row["Popularity"] if pd.notna(row["Popularity"]) else "N/A",
                "Market Share": syn.get("market_share", "N/A"),
                "Credit Score": syn.get("credit_score", "N/A"),
                "Location Rating": syn.get("location_rating", "N/A")
            })

    prospects_df = pd.DataFrame(prospects_list).sort_values(by="Rank")

    # Ensure checkbox state tracking
    if "prospect_checkbox" not in st.session_state:
        st.session_state.prospect_checkbox = {
            name: False for name in prospects_df["Name"]
        }

    # =========================================================
    # 4. Render table INSIDE A FORM → prevents reruns
    # =========================================================
    with st.form("prospects_form_unique"):
        st.subheader("Select Prospects")

        # Header row
        cols = st.columns([1, 3, 3, 2, 2, 2, 2, 2, 1])
        headers = ["Rank", "Name", "Address", "Profit", "Popularity", "Market Share", "Credit Score", "Location Rating", "Select"]
        for c, h in zip(cols, headers):
            c.write(f"**{h}**")

        # Data rows
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

            # Checkbox that does NOT trigger rerun
            state_key = row['Name']
            st.session_state.prospect_checkbox[state_key] = cols[8].checkbox(
                "", value=st.session_state.prospect_checkbox[state_key],
                key=f"prospect_{state_key}"
            )

        submit_button = st.form_submit_button("Generate Selected Leads")

    # =========================================================
    # 5. Handle form submission
    # =========================================================
    if submit_button:
        selected_names = [name for name, selected in st.session_state.prospect_checkbox.items() if selected]

        if selected_names:
            selected_rows = restaurants_df[restaurants_df["Name"].isin(selected_names)]
            current_leads = load_leads_data()
            updated_df = pd.concat([current_leads, selected_rows], ignore_index=True)
            updated_df.to_csv(leads_csv_path, index=False)

            st.success(f"Added {len(selected_names)} prospects as leads.")
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

    assignment_data = []

    if not leads_df.empty and not sales_df.empty:
        for idx, lead in leads_df.iterrows():
            business_name = lead['Name']
            expertise_needed = "Off-Grid Solutions"

            # Get the recommended salesperson (your existing function)
            recommended_salesperson = get_salesperson_recommendation(
                business_name, expertise_needed, leads_df, sales_df
            )

            if recommended_salesperson:
                assignment_data.append({
                    "Business Name": business_name,
                    "Location": lead['Address'],
                    "Sales Person ID": recommended_salesperson.get("Sales Person ID", "N/A"),
                    "Sales Person Name": recommended_salesperson.get("Name", "N/A"),
                    "Sales Person Location": recommended_salesperson.get("Location", "N/A"),
                    "Expertise": recommended_salesperson.get("Expertise", "N/A"),
                    "Experience": recommended_salesperson.get("Experience", "N/A")
                })
            else:
                st.warning(f"No salesperson recommendation for {business_name}")

        if assignment_data:
            assignment_df = pd.DataFrame(assignment_data)

            # SAVE assignments so the Sales Email tab can read them
            assignment_df.to_csv("assignments.csv", index=False)

            st.success("Assignments updated successfully.")
            st.dataframe(assignment_df)
        else:
            st.warning("No assignments were generated.")


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




if tab_selection == "Sales Email":
    st.title("Sales Email Generator")

    # Load leads
    leads_df = load_leads_data()
    sales_df = load_salesperson_data()

    if leads_df.empty:
        st.warning("No leads available. Please generate leads first.")
    else:
        # Lead selection dropdown
        selected_lead = st.selectbox(
            "Select a Lead",
            leads_df["Name"].tolist()
        )

        # Basic lead info
        lead_info = leads_df[leads_df["Name"] == selected_lead].iloc[0]
        business_address = lead_info.get("Address", "Not Available")

        # --------------------------------------------------------------------
        # Load assigned salesperson (from Assignment tab)
        # --------------------------------------------------------------------
        salesperson_name = ""
        salesperson_experience = ""
        salesperson_expertise = "Off-Grid Gas Solutions"
        salesperson_location = ""

        if os.path.exists("assignments.csv"):
            assignments_df = pd.read_csv("assignments.csv")
            lead_assignment = assignments_df[assignments_df["Business Name"] == selected_lead]

            if not lead_assignment.empty:
                salesperson_name = lead_assignment.iloc[0]["Sales Person Name"]
                salesperson_experience = lead_assignment.iloc[0]["Experience"]
                salesperson_expertise = "Off-Grid Gas Solutions"
                salesperson_location = lead_assignment.iloc[0]["Sales Person Location"]
            else:
                st.warning("No assigned salesperson found. Selecting a random one.")
                sp = sales_df.sample(1).iloc[0]
                salesperson_name = sp["Name"]
                salesperson_experience = sp["Experience (Years)"]
                salesperson_expertise = "Off-Grid Gas Solutions"
                salesperson_location = sp["Location (City in Italy)"]
        else:
            st.warning("Assignments file not found. Selecting random salesperson.")
            sp = sales_df.sample(1).iloc[0]
            salesperson_name = sp["Name"]
            salesperson_experience = sp["Experience (Years)"]
            salesperson_expertise = "Off-Grid Gas Solutions"
            salesperson_location = sp["Location (City in Italy)"]

        # --------------------------------------------------------------------
        # Fetch AI-generated business summary (context)
        # --------------------------------------------------------------------
        def get_business_summary(business_name, business_address):
            prompt = f"""
            Provide a short 2–3 sentence description about the business '{business_name}', located at '{business_address}'.
            Highlight what the business is known for and any characteristics relevant to energy usage (such as customer volume, operational hours, food service, hospitality, or location).
            Write in natural, human-like sentences appropriate for a sales email introduction.
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
                    return response.json()["choices"][0]["message"]["content"].strip()
                return ""
            except:
                return ""

        with st.spinner(f"Gathering background for {selected_lead}..."):
            business_context = get_business_summary(selected_lead, business_address)

        # --------------------------------------------------------------------
        # Generate the personalized email with OFF-GRID GAS focus
        # --------------------------------------------------------------------
        def generate_sales_email():
            prompt = f"""
            Write a professional outreach email from SHV Energy addressed to the owners or managers of '{selected_lead}'.

            The email MUST clearly promote SHV Energy’s **Off-Grid Gas solutions**, emphasizing:
            - reliable and uninterrupted gas supply for businesses operating off the main energy grid
            - safer, cleaner alternatives to diesel or old heating systems
            - consistent heating and cooking performance for restaurants / hospitality / food service operations
            - cost stability, high efficiency, and reduced emissions with modern LPG-based systems

            The sender details to include:
            - Name: {salesperson_name}
            - Location: based in {salesperson_location}
            - Experience: {salesperson_experience} years of experience
            - Expertise: Off-Grid Gas Solutions

            Incorporate the following business context naturally at the beginning:
            {business_context}

            Additional instructions:
            - Write in clear, polished paragraphs (3–5 paragraphs).
            - Keep the tone professional, helpful, and confident.
            - Invite the business to a meeting or call to discuss Off-Grid Gas options.
            - Make the benefits specific and relevant to their business type.
            - End with a warm sign-off and a proper email signature for the salesperson.
            - DO NOT mention AI or that this text is generated.

            Output only the email content, no extra commentary.
            """

            headers = {
                "Authorization": f"Bearer {AZURE_OPENAI_API_KEY}",
                "Content-Type": "application/json",
            }
            body = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 700,
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
                    return response.json()["choices"][0]["message"]["content"].strip()
            except:
                pass

            return None

        # Generate email
        with st.spinner("Generating personalized Off-Grid Gas email..."):
            email_output = generate_sales_email()

        if email_output:
            st.markdown("### ✉️ Generated Sales Email (Off-Grid Gas Focused)")
            st.markdown(email_output)

            if st.button("Copy to Clipboard"):
                pyperclip.copy(email_output)
                st.success("Email copied!")
        else:
            st.warning("Failed to generate email.")


if tab_selection == "Carbon Intensity Data":
    st.title("Carbon Intensity Based on AI-Extracted Location")

    # 1️⃣ Load leads from the leads.csv file
    try:
        leads_df = pd.read_csv("leads.csv")  # Changed to leads.csv
    except:
        st.error("Leads file not found. Ensure leads.csv is available.")
        st.stop()

    # Strip spaces from column names just in case
    leads_df.columns = leads_df.columns.str.strip()

    # 2️⃣ User selects a business lead
    selected_business = st.selectbox(
        "Select Business Lead",
        leads_df["Name"].tolist(),  # Changed to "Name" based on your CSV
        index=0  # Default to the first lead
    )

    # Fetch the lead row from the dataframe
    lead_row = leads_df[leads_df["Name"] == selected_business].iloc[0]  # Changed to "Name"
    full_lead_record = lead_row.to_dict()

    # 3️⃣ Load ENV vars for Azure OpenAI (the same you use in Assignment tab)
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

    # ------------------------
    # AI Prompt Construction
    # ------------------------
    prompt = f"""
    You are an expert in Italian geography and Electricity Maps API zones.

    Here is a complete business lead record:
    {full_lead_record}

    TASKS:
    1. Extract the EXACT Italian city or regional location from the address.
    2. Convert that location into the correct Electricity Maps zone code.
    3. Return data ONLY in this exact JSON format:

    {{
        "location": "[Extracted Italian location]",
        "zone": "[Electricity Maps zone code]"
    }}

    Examples:
    Central North Italy -> IT-CNO
    Central South Italy -> IT-CSO
    North Italy -> IT-NO
    Sardinia -> IT-SAR
    Sicily -> IT-SIC
    South Italy -> IT-SO

    DO NOT RETURN ANYTHING OTHER THAN THE JSON ABOVE.
    """

    # ------------------------
    # Prepare request to Azure OpenAI
    # ------------------------
    headers = {
        "Authorization": f"Bearer {AZURE_OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "temperature": 0.2,
    }

    try:
        ai_response = requests.post(
            f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}",
            headers=headers,
            json=body,
            timeout=30,
        )

        if ai_response.status_code != 200:
            st.error(f"Azure OpenAI Error {ai_response.status_code}: {ai_response.text}")
            st.stop()

        ai_content = ai_response.json()["choices"][0]["message"]["content"].strip()

        try:
            location_obj = json.loads(ai_content)
        except json.JSONDecodeError:
            st.error("AI response is not valid JSON:")
            st.write(ai_content)
            st.stop()

        location = location_obj.get("location")
        zone = location_obj.get("zone")

        st.success(f"📍 AI Extracted Location: {location}")
        st.success(f"🔌 Electricity Maps Zone: {zone}")

    except Exception as e:
        st.error(f"Error calling Azure OpenAI: {e}")
        st.stop()

    # ------------------------
    # Zone Mapping Based on Region Name
    # ------------------------
    zone_mapping = {
        "Central North Italy": "IT-CNO",
        "Central South Italy": "IT-CSO",
        "North Italy": "IT-NO",
        "Sardinia": "IT-SAR",
        "Sicily": "IT-SIC",
        "South Italy": "IT-SO",
    }

    # If zone is not provided by AI, try mapping the region
    if not zone:
        zone = zone_mapping.get(location, None)

    # If no valid zone is found, display an error
    if not zone:
        st.error(f"Could not map region '{location}' to a valid Electricity Maps zone.")
        st.stop()

    st.success(f"Resolved Zone: {zone}")

    # ------------------------
    # Make the API Call
    # ------------------------
    carbon_url = "https://api.electricitymaps.com/v3/carbon-intensity/latest"
    headers = {"auth-token": os.getenv("ELECTRICITYMAPS_API_KEY")}  # Ensure correct header for the API
    params = {"zone": zone}

    resp = requests.get(carbon_url, headers=headers, params=params)

    if resp.status_code == 200:
        carbon = resp.json()

        st.subheader(f"🌍 Carbon Intensity for {location} ({zone})")
        st.write(f"**Current Carbon Intensity:** {carbon['carbonIntensity']} gCO₂/kWh")

        # ------------------------
        # Convert API response to a table format
        # ------------------------
        carbon_data = {
            "Field": [
                "Zone",
                "Carbon Intensity (gCO₂/kWh)",
                "Datetime",
                "Updated At",
                "Created At",
                "Emission Factor Type",
                "Is Estimated",
                "Estimation Method",
                "Temporal Granularity",
                "_Disclaimer"
            ],
            "Value": [
                carbon.get("zone", "N/A"),
                carbon.get("carbonIntensity", "N/A"),
                carbon.get("datetime", "N/A"),
                carbon.get("updatedAt", "N/A"),
                carbon.get("createdAt", "N/A"),
                carbon.get("emissionFactorType", "N/A"),
                carbon.get("isEstimated", "N/A"),
                carbon.get("estimationMethod", "N/A"),
                carbon.get("temporalGranularity", "N/A"),
                carbon.get("_disclaimer", "N/A")
            ]
        }

        st.table(pd.DataFrame(carbon_data))

    else:
        st.error(f"Electricity Maps API Error {resp.status_code}: {resp.text}")


if tab_selection == "Additional Insights":

    st.title("Additional Insights for Enhanced Lead Profile")

    st.markdown("### 🇮🇹 Italy – Open Data Maturity (ODM) 2024 Overview")

    # -------------- TABLE 1: MATURITY SCORES --------------------
    st.subheader("📊 Open Data Maturity Scores (2024)")

    maturity_data = {
        "Dimension": ["Policy", "Portal", "Quality", "Impact"],
        "Score": [93.8, 89.6, 85.7, 100],
        "Year-on-Year Change": ["+2.0%", "+3.4%", "+1.3%", "+5.8%"],
    }

    maturity_df = pd.DataFrame(maturity_data)
    st.table(maturity_df)

    # ---------------- KEY QUESTIONS -----------------------
    st.markdown("### ❓ Key Questions for Each Dimension")

    insights = {
        "Policy Dimension": [
            "Does the national open data policy/strategy include an action plan? – **Yes**",
            "To what degree do local/regional bodies run open data initiatives? – **All**",
            "Are there processes to ensure policies are implemented? – **Yes**",
        ],
        "Portal Dimension": [
            "Do you monitor the portal’s traffic? – **Yes**",
            "Do public sector providers contribute data? – **About half**",
            "Does the portal list existing but non-open datasets? – **Yes**",
        ],
        "Quality Dimension": [
            "Do you monitor metadata quality? – **Yes**",
            "Do providers have to follow metadata standards? – **Yes**",
            "Is there a model to assess data quality? – **Yes**",
        ],
        "Impact Dimension": [
            "Is there a definition of open data reuse? – **Yes**",
            "Are there processes to monitor reuse? – **Yes**",
            "Do you measure the impact of open data? – **Yes**",
        ],
    }

    for section, q_list in insights.items():
        st.subheader(section)
        for q in q_list:
            st.write(f"- {q}")

    # ---------------- PDF PREVIEW -----------------------
    st.markdown("### 📄 Reference: ODM 2024 Italy Factsheet")
    st.write("Below is a preview of the uploaded PDF used for insights:")

    try:
        with open("2024_odm_factsheet_italy.pdf", "rb") as pdf_file:
            st.download_button(
                label="📥 Download Italy ODM 2024 Factsheet PDF",
                data=pdf_file,
                file_name="2024_odm_factsheet_italy.pdf",
                mime="application/pdf"
            )
        st.info("PDF Loaded Successfully. Showing preview image…")
    except Exception as e:
        st.error(f"Could not load the PDF: {e}")
if tab_selection == "Targeted Marketing Strategy":

    st.title("🎯 Targeted Marketing Strategy")

    # ------------------------------------------------------------------
    # Load the assignments (lead + salesperson combined profile)
    # ------------------------------------------------------------------
    try:
        assignment_df = pd.read_csv("assignments.csv")
    except:
        st.error("Assignments not found. Please run Assignment tab first.")
        st.stop()

    # Select lead to generate strategy for
    selected_business = st.selectbox(
        "Select Business Lead",
        assignment_df["Business Name"].tolist()
    )

    # Extract full combined profile for this lead
    lead_data = assignment_df[assignment_df["Business Name"] == selected_business].iloc[0].to_dict()

    # Try displaying nicely
    with st.expander("View Lead Profile Used for Strategy"):
        st.json(lead_data)

    # ------------------------------------------------------------------
    # OPTIONAL: Pull the Carbon Intensity if stored in your assignment df
    # ------------------------------------------------------------------
    carbon_intensity = lead_data.get("CarbonIntensity", None)

    # Load ODM PDF (context)
    odm_pdf_path = "2024_odm_factsheet_italy.pdf"
    odm_context = """
    Italy's Open Data Maturity (ODM) 2024 indicates a highly advanced national digital ecosystem.

    Key quick facts used for marketing context:
    - Policy: 93.8  
    - Portal: 89.6  
    - Quality: 85.7  
    - Impact: 100  
    - Strong adoption of open data for local & regional development  
    - Strong public initiatives supporting sustainability and green innovation
    """

    # ------------------------------------------------------------------
    # Azure OpenAI ENV
    # ------------------------------------------------------------------
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

    # ------------------------------------------------------------------
    # Build the prompt to send ALL RELEVANT INFO to OpenAI
    # ------------------------------------------------------------------

    prompt = f"""
    You are an expert in marketing strategy, off-grid gas solutions, Italian SME behavior, 
    and regional energy economics.

    Create a **deeply personalized marketing strategy** for the exact restaurant lead below.

    ===============================
    LEAD PROFILE (FULL DATA INPUT)
    ===============================
    {json.dumps(lead_data, indent=2)}

    ===============================
    CARBON INTENSITY (IF AVAILABLE)
    ===============================
    {carbon_intensity}

    ===============================
    ITALY OPEN DATA MATURITY (ODM 2024)
    USED TO UNDERSTAND ECONOMIC CONTEXT
    ===============================
    {odm_context}

    ===============================
    YOUR TASK
    ===============================

    Create a **highly tailored marketing strategy** ONLY for this specific lead.  
    Base all reasoning strictly on the provided data.  
    Do NOT generalize.  
    Write the output in beautifully formatted MARKDOWN.

    Include the following sections:

    1. **Lead Summary**
    2. **Operational Pain Points (Based on location, type, popularity, profit)**
    3. **Energy Risk Profile (Using carbon intensity if provided)**
    4. **Tailored Off-Grid Gas Solution Recommendation**
    5. **Financial ROI Estimation (based strictly on lead data)**
    6. **Environmental Impact & Emission Benefits**
    7. **Recommended Messaging Style (tone, angle)**
    8. **Targeted Outreach Plan (step-by-step)**
    9. **Campaign Ideas (local incentives, context from ODM Italy)**

    Make everything look clean with bold headers, bullet points, and spacing.
    """

    # ------------------------------------------------------------------
    # CALL AZURE OPENAI
    # ------------------------------------------------------------------

    headers = {
        "Authorization": f"Bearer {AZURE_OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500,
        "temperature": 0.3,
    }

    # Using a spinner to indicate the request is in progress
    with st.spinner("Generating marketing strategy..."):
        try:
            response = requests.post(
                f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}",
                headers=headers,
                json=body,
                timeout=45,
            )

            if response.status_code != 200:
                st.error(f"OpenAI Error {response.status_code}: {response.text}")
                st.stop()

            final_strategy = response.json()["choices"][0]["message"]["content"]

        except Exception as e:
            st.error(f"OpenAI request failed: {e}")
            st.stop()

    # ------------------------------------------------------------------
    # DISPLAY THE FINAL MARKETING STRATEGY
    # ------------------------------------------------------------------

    st.subheader(f"📌 Personalized Marketing Strategy for {selected_business}")
    st.markdown(final_strategy)

    st.markdown("---")
    st.info("This strategy is automatically generated using Azure OpenAI using all relevant lead data, carbon intensity, and Italian economic insights.")



