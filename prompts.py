prompt_0 = """
# TODO: complete the following function in one line. It should give the answer to: How many rows are there in this dataframe? 
def example(df: pd.DataFrame) -> int:
    df.columns=["A"]
    return df.shape[0]

# TODO: complete the following function in one line. It should give the answer to: {question}
def answer(df: pd.DataFrame) -> {row_type}:
    df.columns = {list_df_columns}
    return"""

prompt_1 = """
Given a pandas DataFrame operation task, complete the function in one line.

Input DataFrame columns: {list_df_columns}
Question: {question}
Expected return type: {row_type}

Example format:
def answer(df: pd.DataFrame) -> float:
    df.columns = ['A', 'B', 'C']
    return ...

Write a single line of code to replace the ... that:
1. Uses the DataFrame 'df' as input
2. Returns the exact type specified in the return type
3. Solves the given question
4. Uses pandas operations efficiently
5. Avoids loops or multiple statements

The solution should be ONLY one line and use pandas vectorized operations when possible. 

Format your output as follows:
```json
{{
    "solution": "your one line of code here"
}}
```
"""

prompt_2 = """You are an expert data scientist working at a large company. You have been tasked with writing a function that will gather insights from a dataset. The function should take a pandas DataFrame as input and return the requested information. 
Given the following input DataFrame columns, question, and expected return type, complete the function in one line.

Input DataFrame columns: {list_df_columns}
Question: {question}
Expected return type: {row_type}

Example format:
def answer(df: pd.DataFrame) -> float:
    df.columns = ['A', 'B', 'C']
    return ...

Write a single line of code to replace the ... that:
1. Uses the DataFrame 'df' as input
2. Returns the exact type specified in the return type
3. Solves the given question
4. Uses pandas operations efficiently
5. Avoids loops or multiple statements

The solution should be ONLY one line and use pandas vectorized operations when possible. 

Format your output as follows:
```json
{{
    "solution": "your one line of code here"
}}
```
"""

prompt_3 = """You are an expert data scientist working at a large company. You have been tasked with writing a function that will gather insights from a dataset. The function should take a pandas DataFrame as input and return the requested information. 
Given the following input DataFrame columns, question, and expected return type, complete the function in one line.

Input DataFrame columns: {list_df_columns}
Question: {question}
Expected return type: {row_type}

Example format:
def answer(df: pd.DataFrame) -> float:
    df.columns = ['A', 'B', 'C']
    return ...

Write a single line of code to replace the ... that:
1. Uses the DataFrame 'df' as input
2. Returns the exact type specified in the return type
3. Solves the given question
4. Uses pandas operations efficiently
5. Avoids loops or multiple statements

The solution should be ONLY one line and use pandas vectorized operations when possible. 

Format your output as follows:
```json
{{
    "solution": "your one line of code here"
}}
```

Here are a few examples:

Example 1:

Input DataFrame columns: ['rank', 'personName', 'age', 'finalWorth', 'category', 'source', 'country', 'state', 'city', 'organization', 'selfMade', 'gender', 'birthDate', 'title', 'philanthropyScore', 'bio', 'about']
Question: Is the person with the highest net worth self-made?
Expected return type: boolean

Output:
```json
{{
    "solution": "df.loc[df['finalWorth'].idxmax(), 'selfMade']"
}}
```

Example 2:

Input DataFrame columns: ['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'Siblings_Spouses Aboard', 'Parents_Children Aboard', 'Fare']
Question: Which passenger class has the highest number of survivors?
Expected return type: category

Output:
```json
{{
    "solution": "df[df['Survived'] == 1]['Pclass'].mode().iloc[0]\nans = answer(df)"
}}
```

Now it's your turn! Give me the output for the following:
Input DataFrame columns: {list_df_columns}
Question: {question}
Expected return type: {row_type}

Output:
"""

prompt_4 = """You are an expert data scientist working at a large company. You have been tasked with writing a function that will gather insights from a dataset. The function should take a pandas DataFrame as input and return the requested information. 
Given the following input DataFrame columns, question, and expected return type, complete the function in one line.

Input DataFrame columns: {list_df_columns}
Question: {question}
Expected return type: {row_type}

Example format:
def answer(df: pd.DataFrame) -> float:
    df.columns = ['A', 'B', 'C']
    return ...

Write a single line of code to replace the ... that:
1. Uses the DataFrame 'df' as input
2. Returns the exact type specified in the return type
3. Solves the given question
4. Uses pandas operations efficiently
5. Avoids loops or multiple statements

The solution should be ONLY one line and use pandas vectorized operations when possible. 

Format your output as follows:
```json
{{
    "solution": "your one line of code here"
}}
```

Here are a few examples:

Example 1:

Input DataFrame columns: ['rank', 'personName', 'age', 'finalWorth', 'category', 'source', 'country', 'state', 'city', 'organization', 'selfMade', 'gender', 'birthDate', 'title', 'philanthropyScore', 'bio', 'about']
Question: Is the person with the highest net worth self-made?
Expected return type: boolean

Output:
```json
{{
    "solution": "df.loc[df['finalWorth'].idxmax(), 'selfMade']"
}}
```

Example 2:

Input DataFrame columns: ['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'Siblings_Spouses Aboard', 'Parents_Children Aboard', 'Fare']
Question: Which passenger class has the highest number of survivors?
Expected return type: category

Output:
```json
{{
    "solution": "df[df['Survived'] == 1]['Pclass'].mode().iloc[0]"
}}
```

Example 3:

Input DataFrame columns: ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
Question: Are there any individuals in the dataset who are above 60 years of age?
Expected return type: boolean

Output:
```json
{{
    "solution": "return df['Age'].gt(60).any()"
}}
```

Example 4:

Input DataFrame columns: ['Submitted at', 'What is your age? \ud83d\udc76\ud83c\udffb\ud83d\udc75\ud83c\udffb', \"What's your nationality?\", 'What is your civil status? \ud83d\udc8d', \"What's your sexual orientation?\", 'Do you have children? \ud83c\udf7c', 'What is the maximum level of studies you have achieved? \ud83c\udf93', 'Gross annual salary (in euros) \ud83d\udcb8', \"What's your height? in cm \ud83d\udccf\", \"What's your weight? in Kg \u2696\ufe0f\", 'What is your body complexity? \ud83c\udfcb\ufe0f', 'What is your eye color? \ud83d\udc41\ufe0f', 'What is your hair color? \ud83d\udc69\ud83e\uddb0\ud83d\udc71\ud83c\udffd', 'What is your skin tone?', 'How long is your hair? \ud83d\udc87\ud83c\udffb\u2640\ufe0f\ud83d\udc87\ud83c\udffd\u2642\ufe0f', 'How long is your facial hair? \ud83e\uddd4\ud83c\udffb', 'How often do you wear glasses? \ud83d\udc53', 'How attractive do you consider yourself?', 'Have you ever use an oline dating app?', 'Where have you met your sexual partners? (In a Bar or Restaurant)', 'Where have you met your sexual partners? (Through Friends)', 'Where have you met your sexual partners? (Through Work or as Co-Workers)', 'Where have you met your sexual partners? (Through Family)', 'Where have you met your sexual partners? (in University)', 'Where have you met your sexual partners? (in Primary or Secondary School)', 'Where have you met your sexual partners? (Neighbors)', 'Where have you met your sexual partners? (in Church)', 'Where have you met your sexual partners? (Other)', 'How many people have you kissed?', 'How many sexual partners have you had?', 'How many people have you considered as your boyfriend_girlfriend?', 'How many times per month did you practice sex lately?', 'Happiness scale', 'What area of knowledge is closer to you?', 'If you are in a relationship, how long have you been with your partner?']
Question: What are the top 4 maximum gross annual salaries?
Expected return type: list[number]

Output:
```json
{{
    "solution": "df['Gross annual salary (in euros) \ud83d\udcb8'].nlargest(4).tolist()"
}}
```

Example 5:

Input DataFrame columns: ['segmentation_1', 'descriptor', 'complaint_type', 'created_date', 'borough', 'hour', 'month_name', 'weekday_name', 'agency', 'resolution_description', 'agency_name', 'city', 'location_type', 'incident_zip', 'incident_address', 'street_name', 'cross_street_1', 'cross_street_2', 'intersection_street_1', 'intersection_street_2', 'address_type', 'landmark', 'facility_type', 'status', 'due_date', 'resolution_action_updated_date', 'community_board', 'x_coordinate', 'y_coordinate', 'park_facility_name', 'park_borough', 'bbl', 'open_data_channel_type', 'vehicle_type', 'taxi_company_borough', 'taxi_pickup_location', 'bridge_highway_name', 'bridge_highway_direction', 'road_ramp', 'bridge_highway_segment', 'latitude', 'longitude', 'location', 'unique_key', 'Unnamed_0', 'closed_date']
Question: Which 4 agencies handle the most complaints?
Expected return type: list[category]

Output:
```json
{{
    "solution": "df['agency'].value_counts().nlargest(4).index.tolist()"
}}
```

Example 6:

Input DataFrame columns: ['yr', 'mo', 'dy', 'date', 'st', 'mag', 'inj', 'fat', 'slat', 'slon', 'elat', 'elon', 'len', 'wid']
Question: What is the maximum number of injuries caused by a single tornado?
Expected return type: number

Output:
```json
{{
    "solution": "df['inj'].max()"
}}
```

Example 7:

Input DataFrame columns: ['ID', 'Year_Birth', 'Education', 'Marital_Status', 'Income', 'Kidhome', 'Teenhome', 'Dt_Customer', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Z_CostContact', 'Z_Revenue', 'Response']
Question: Is there any customer with income higher than 100000?
Expected return type: boolean

Output:
```json
{{
    "solution": "df['Income'].gt(100000).any()"
}}
```

Example 8:

Input DataFrame columns: ['neighbourhood_cleansed', 'host_neighbourhood', 'price', 'room_type', 'reviews_per_month', 'property_type', 'bedrooms', 'host_verifications', 'host_acceptance_rate', 'host_identity_verified', 'beds', 'amenities', 'minimum_nights', 'last_review', 'review_scores_rating', 'instant_bookable', 'calculated_host_listings_count', 'first_review', 'number_of_reviews', 'accommodates', 'listing_url', 'last_scraped', 'source', 'name', 'description', 'neighborhood_overview', 'picture_url', 'host_id', 'host_name', 'host_since', 'host_location', 'host_about', 'host_response_time', 'host_response_rate', 'host_is_superhost', 'host_thumbnail_url', 'host_picture_url', 'host_listings_count', 'host_total_listings_count', 'host_has_profile_pic', 'neighbourhood', 'latitude', 'longitude', 'bathrooms', 'bathrooms_text', 'maximum_nights', 'minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights', 'maximum_maximum_nights', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'calendar_updated', 'has_availability', 'availability_30', 'availability_60', 'availability_90', 'availability_365', 'calendar_last_scraped', 'number_of_reviews_ltm', 'number_of_reviews_l30d', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 'license', 'calculated_host_listings_count_entire_homes', 'calculated_host_listings_count_private_rooms', 'calculated_host_listings_count_shared_rooms', 'x', 'y', 'price_M0jI']\nQuestion
Question: Which host verification method is the least used?
Expected return type: category

Output:
```json
{{
    "solution": "df['host_verifications'].str.split(', ').explode().value_counts().idxmin()"
}}
```

Now it's your turn! Give me the output for the following:
Input DataFrame columns: {list_df_columns}
Question: {question}
Expected return type: {row_type}

Output:
"""

prompt_5 = """Input DataFrame columns: ['rank', 'personName', 'age', 'finalWorth', 'category', 'source', 'country', 'state', 'city', 'organization', 'selfMade', 'gender', 'birthDate', 'title', 'philanthropyScore', 'bio', 'about']
Question: Is the person with the highest net worth self-made?
Expected return type: boolean

Code that calculates the answer:
```json
{{
    "solution": "df.loc[df['finalWorth'].idxmax(), 'selfMade']"
}}

Input DataFrame columns: ['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'Siblings_Spouses Aboard', 'Parents_Children Aboard', 'Fare']
Question: Which passenger class has the highest number of survivors?
Expected return type: category

Code that calculates the answer:
```json
{{
    "solution": "df[df['Survived'] == 1]['Pclass'].mode().iloc[0]"
}}
```

Input DataFrame columns: ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
Question: Are there any individuals in the dataset who are above 60 years of age?
Expected return type: boolean

Code that calculates the answer:
```json
{{
    "solution": "return df['Age'].gt(60).any()"
}}
```

Input DataFrame columns: ['Submitted at', 'What is your age? \ud83d\udc76\ud83c\udffb\ud83d\udc75\ud83c\udffb', \"What's your nationality?\", 'What is your civil status? \ud83d\udc8d', \"What's your sexual orientation?\", 'Do you have children? \ud83c\udf7c', 'What is the maximum level of studies you have achieved? \ud83c\udf93', 'Gross annual salary (in euros) \ud83d\udcb8', \"What's your height? in cm \ud83d\udccf\", \"What's your weight? in Kg \u2696\ufe0f\", 'What is your body complexity? \ud83c\udfcb\ufe0f', 'What is your eye color? \ud83d\udc41\ufe0f', 'What is your hair color? \ud83d\udc69\ud83e\uddb0\ud83d\udc71\ud83c\udffd', 'What is your skin tone?', 'How long is your hair? \ud83d\udc87\ud83c\udffb\u2640\ufe0f\ud83d\udc87\ud83c\udffd\u2642\ufe0f', 'How long is your facial hair? \ud83e\uddd4\ud83c\udffb', 'How often do you wear glasses? \ud83d\udc53', 'How attractive do you consider yourself?', 'Have you ever use an oline dating app?', 'Where have you met your sexual partners? (In a Bar or Restaurant)', 'Where have you met your sexual partners? (Through Friends)', 'Where have you met your sexual partners? (Through Work or as Co-Workers)', 'Where have you met your sexual partners? (Through Family)', 'Where have you met your sexual partners? (in University)', 'Where have you met your sexual partners? (in Primary or Secondary School)', 'Where have you met your sexual partners? (Neighbors)', 'Where have you met your sexual partners? (in Church)', 'Where have you met your sexual partners? (Other)', 'How many people have you kissed?', 'How many sexual partners have you had?', 'How many people have you considered as your boyfriend_girlfriend?', 'How many times per month did you practice sex lately?', 'Happiness scale', 'What area of knowledge is closer to you?', 'If you are in a relationship, how long have you been with your partner?']
Question: What are the top 4 maximum gross annual salaries?
Expected return type: list[number]

Code that calculates the answer:
```json
{{
    "solution": "df['Gross annual salary (in euros) \ud83d\udcb8'].nlargest(4).tolist()"
}}
```

Input DataFrame columns: ['segmentation_1', 'descriptor', 'complaint_type', 'created_date', 'borough', 'hour', 'month_name', 'weekday_name', 'agency', 'resolution_description', 'agency_name', 'city', 'location_type', 'incident_zip', 'incident_address', 'street_name', 'cross_street_1', 'cross_street_2', 'intersection_street_1', 'intersection_street_2', 'address_type', 'landmark', 'facility_type', 'status', 'due_date', 'resolution_action_updated_date', 'community_board', 'x_coordinate', 'y_coordinate', 'park_facility_name', 'park_borough', 'bbl', 'open_data_channel_type', 'vehicle_type', 'taxi_company_borough', 'taxi_pickup_location', 'bridge_highway_name', 'bridge_highway_direction', 'road_ramp', 'bridge_highway_segment', 'latitude', 'longitude', 'location', 'unique_key', 'Unnamed_0', 'closed_date']
Question: Which 4 agencies handle the most complaints?
Expected return type: list[category]

Code that calculates the answer:
```json
{{
    "solution": "df['agency'].value_counts().nlargest(4).index.tolist()"
}}
```

Input DataFrame columns: ['yr', 'mo', 'dy', 'date', 'st', 'mag', 'inj', 'fat', 'slat', 'slon', 'elat', 'elon', 'len', 'wid']
Question: What is the maximum number of injuries caused by a single tornado?
Expected return type: number

Code that calculates the answer:
```json
{{
    "solution": "df['inj'].max()"
}}
```

Input DataFrame columns: ['ID', 'Year_Birth', 'Education', 'Marital_Status', 'Income', 'Kidhome', 'Teenhome', 'Dt_Customer', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Z_CostContact', 'Z_Revenue', 'Response']
Question: Is there any customer with income higher than 100000?
Expected return type: boolean

Code that calculates the answer:
```json
{{
    "solution": "df['Income'].gt(100000).any()"
}}
```

Input DataFrame columns: ['neighbourhood_cleansed', 'host_neighbourhood', 'price', 'room_type', 'reviews_per_month', 'property_type', 'bedrooms', 'host_verifications', 'host_acceptance_rate', 'host_identity_verified', 'beds', 'amenities', 'minimum_nights', 'last_review', 'review_scores_rating', 'instant_bookable', 'calculated_host_listings_count', 'first_review', 'number_of_reviews', 'accommodates', 'listing_url', 'last_scraped', 'source', 'name', 'description', 'neighborhood_overview', 'picture_url', 'host_id', 'host_name', 'host_since', 'host_location', 'host_about', 'host_response_time', 'host_response_rate', 'host_is_superhost', 'host_thumbnail_url', 'host_picture_url', 'host_listings_count', 'host_total_listings_count', 'host_has_profile_pic', 'neighbourhood', 'latitude', 'longitude', 'bathrooms', 'bathrooms_text', 'maximum_nights', 'minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights', 'maximum_maximum_nights', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'calendar_updated', 'has_availability', 'availability_30', 'availability_60', 'availability_90', 'availability_365', 'calendar_last_scraped', 'number_of_reviews_ltm', 'number_of_reviews_l30d', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 'license', 'calculated_host_listings_count_entire_homes', 'calculated_host_listings_count_private_rooms', 'calculated_host_listings_count_shared_rooms', 'x', 'y', 'price_M0jI']\nQuestion
Question: Which host verification method is the least used?
Expected return type: category

Code that calculates the answer:
```json
{{
    "solution": "df['host_verifications'].str.split(', ').explode().value_counts().idxmin()"
}}
```

Input DataFrame columns: {list_df_columns}
Question: {question}
Expected return type: {row_type}

Code that calculates the answer:
"""

prompt_6 = """You are an expert data scientist working at a large company. You have been tasked with writing a function that will gather insights from a dataset. The function should take a pandas DataFrame as input and return the requested information. 
Given the following input DataFrame columns, question, and expected return type, complete the function in one line.
The solution should be ONLY one line and use pandas vectorized operations when possible. 

Input DataFrame columns: ['rank', 'personName', 'age', 'finalWorth', 'category', 'source', 'country', 'state', 'city', 'organization', 'selfMade', 'gender', 'birthDate', 'title', 'philanthropyScore', 'bio', 'about']
Question: Is the person with the highest net worth self-made?
Expected return type: boolean

Code that calculates the answer:
```json
{{
    "solution": "df.loc[df['finalWorth'].idxmax(), 'selfMade']"
}}

Input DataFrame columns: ['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'Siblings_Spouses Aboard', 'Parents_Children Aboard', 'Fare']
Question: Which passenger class has the highest number of survivors?
Expected return type: category

Code that calculates the answer:
```json
{{
    "solution": "df[df['Survived'] == 1]['Pclass'].mode().iloc[0]"
}}
```

Input DataFrame columns: ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
Question: Are there any individuals in the dataset who are above 60 years of age?
Expected return type: boolean

Code that calculates the answer:
```json
{{
    "solution": "return df['Age'].gt(60).any()"
}}
```

Input DataFrame columns: ['Submitted at', 'What is your age? \ud83d\udc76\ud83c\udffb\ud83d\udc75\ud83c\udffb', \"What's your nationality?\", 'What is your civil status? \ud83d\udc8d', \"What's your sexual orientation?\", 'Do you have children? \ud83c\udf7c', 'What is the maximum level of studies you have achieved? \ud83c\udf93', 'Gross annual salary (in euros) \ud83d\udcb8', \"What's your height? in cm \ud83d\udccf\", \"What's your weight? in Kg \u2696\ufe0f\", 'What is your body complexity? \ud83c\udfcb\ufe0f', 'What is your eye color? \ud83d\udc41\ufe0f', 'What is your hair color? \ud83d\udc69\ud83e\uddb0\ud83d\udc71\ud83c\udffd', 'What is your skin tone?', 'How long is your hair? \ud83d\udc87\ud83c\udffb\u2640\ufe0f\ud83d\udc87\ud83c\udffd\u2642\ufe0f', 'How long is your facial hair? \ud83e\uddd4\ud83c\udffb', 'How often do you wear glasses? \ud83d\udc53', 'How attractive do you consider yourself?', 'Have you ever use an oline dating app?', 'Where have you met your sexual partners? (In a Bar or Restaurant)', 'Where have you met your sexual partners? (Through Friends)', 'Where have you met your sexual partners? (Through Work or as Co-Workers)', 'Where have you met your sexual partners? (Through Family)', 'Where have you met your sexual partners? (in University)', 'Where have you met your sexual partners? (in Primary or Secondary School)', 'Where have you met your sexual partners? (Neighbors)', 'Where have you met your sexual partners? (in Church)', 'Where have you met your sexual partners? (Other)', 'How many people have you kissed?', 'How many sexual partners have you had?', 'How many people have you considered as your boyfriend_girlfriend?', 'How many times per month did you practice sex lately?', 'Happiness scale', 'What area of knowledge is closer to you?', 'If you are in a relationship, how long have you been with your partner?']
Question: What are the top 4 maximum gross annual salaries?
Expected return type: list[number]

Code that calculates the answer:
```json
{{
    "solution": "df['Gross annual salary (in euros) \ud83d\udcb8'].nlargest(4).tolist()"
}}
```

Input DataFrame columns: ['segmentation_1', 'descriptor', 'complaint_type', 'created_date', 'borough', 'hour', 'month_name', 'weekday_name', 'agency', 'resolution_description', 'agency_name', 'city', 'location_type', 'incident_zip', 'incident_address', 'street_name', 'cross_street_1', 'cross_street_2', 'intersection_street_1', 'intersection_street_2', 'address_type', 'landmark', 'facility_type', 'status', 'due_date', 'resolution_action_updated_date', 'community_board', 'x_coordinate', 'y_coordinate', 'park_facility_name', 'park_borough', 'bbl', 'open_data_channel_type', 'vehicle_type', 'taxi_company_borough', 'taxi_pickup_location', 'bridge_highway_name', 'bridge_highway_direction', 'road_ramp', 'bridge_highway_segment', 'latitude', 'longitude', 'location', 'unique_key', 'Unnamed_0', 'closed_date']
Question: Which 4 agencies handle the most complaints?
Expected return type: list[category]

Code that calculates the answer:
```json
{{
    "solution": "df['agency'].value_counts().nlargest(4).index.tolist()"
}}
```

Input DataFrame columns: ['yr', 'mo', 'dy', 'date', 'st', 'mag', 'inj', 'fat', 'slat', 'slon', 'elat', 'elon', 'len', 'wid']
Question: What is the maximum number of injuries caused by a single tornado?
Expected return type: number

Code that calculates the answer:
```json
{{
    "solution": "df['inj'].max()"
}}
```

Input DataFrame columns: ['ID', 'Year_Birth', 'Education', 'Marital_Status', 'Income', 'Kidhome', 'Teenhome', 'Dt_Customer', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Z_CostContact', 'Z_Revenue', 'Response']
Question: Is there any customer with income higher than 100000?
Expected return type: boolean

Code that calculates the answer:
```json
{{
    "solution": "df['Income'].gt(100000).any()"
}}
```

Input DataFrame columns: ['neighbourhood_cleansed', 'host_neighbourhood', 'price', 'room_type', 'reviews_per_month', 'property_type', 'bedrooms', 'host_verifications', 'host_acceptance_rate', 'host_identity_verified', 'beds', 'amenities', 'minimum_nights', 'last_review', 'review_scores_rating', 'instant_bookable', 'calculated_host_listings_count', 'first_review', 'number_of_reviews', 'accommodates', 'listing_url', 'last_scraped', 'source', 'name', 'description', 'neighborhood_overview', 'picture_url', 'host_id', 'host_name', 'host_since', 'host_location', 'host_about', 'host_response_time', 'host_response_rate', 'host_is_superhost', 'host_thumbnail_url', 'host_picture_url', 'host_listings_count', 'host_total_listings_count', 'host_has_profile_pic', 'neighbourhood', 'latitude', 'longitude', 'bathrooms', 'bathrooms_text', 'maximum_nights', 'minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights', 'maximum_maximum_nights', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'calendar_updated', 'has_availability', 'availability_30', 'availability_60', 'availability_90', 'availability_365', 'calendar_last_scraped', 'number_of_reviews_ltm', 'number_of_reviews_l30d', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 'license', 'calculated_host_listings_count_entire_homes', 'calculated_host_listings_count_private_rooms', 'calculated_host_listings_count_shared_rooms', 'x', 'y', 'price_M0jI']\nQuestion
Question: Which host verification method is the least used?
Expected return type: category

Code that calculates the answer:
```json
{{
    "solution": "df['host_verifications'].str.split(', ').explode().value_counts().idxmin()"
}}
```

Input DataFrame columns: {list_df_columns}
Question: {question}
Expected return type: {row_type}

Code that calculates the answer:
"""

prompt_7 = """You are an expert data scientist working at a large company. You have been tasked with writing a function that will gather insights from a dataset. The function should take a pandas DataFrame as input and return the requested information. 
Given the following input DataFrame columns, question, and expected return type, complete the function in one line.
The solution should be ONLY one line and use pandas vectorized operations when possible. 

Input DataFrame columns: ['rank', 'personName', 'age', 'finalWorth', 'category', 'source', 'country', 'state', 'city', 'organization', 'selfMade', 'gender', 'birthDate', 'title', 'philanthropyScore', 'bio', 'about']
Question: Is the person with the highest net worth self-made?
Expected return type: boolean

Code that calculates the answer:
```json
{{
    "solution": "df.loc[df['finalWorth'].idxmax(), 'selfMade']"
}}

Input DataFrame columns: ['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'Siblings_Spouses Aboard', 'Parents_Children Aboard', 'Fare']
Question: Which passenger class has the highest number of survivors?
Expected return type: category

Code that calculates the answer:
```json
{{
    "solution": "df[df['Survived'] == 1]['Pclass'].mode().iloc[0]"
}}
```

Input DataFrame columns: {list_df_columns}
Question: {question}
Expected return type: {row_type}

Code that calculates the answer:
"""

prompt_8 = """Input DataFrame columns: ['rank', 'personName', 'age', 'finalWorth', 'category', 'source', 'country', 'state', 'city', 'organization', 'selfMade', 'gender', 'birthDate', 'title', 'philanthropyScore', 'bio', 'about']
Question: Is the person with the highest net worth self-made?
Expected return type: boolean

Output:
```json
{{
    "solution": "df.loc[df['finalWorth'].idxmax(), 'selfMade']"
}}

Input DataFrame columns: ['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'Siblings_Spouses Aboard', 'Parents_Children Aboard', 'Fare']
Question: Which passenger class has the highest number of survivors?
Expected return type: category

Output:
```json
{{
    "solution": "df[df['Survived'] == 1]['Pclass'].mode().iloc[0]"
}}
```

Input DataFrame columns: ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
Question: Are there any individuals in the dataset who are above 60 years of age?
Expected return type: boolean

Output:
```json
{{
    "solution": "return df['Age'].gt(60).any()"
}}
```

Input DataFrame columns: ['Submitted at', 'What is your age? \ud83d\udc76\ud83c\udffb\ud83d\udc75\ud83c\udffb', \"What's your nationality?\", 'What is your civil status? \ud83d\udc8d', \"What's your sexual orientation?\", 'Do you have children? \ud83c\udf7c', 'What is the maximum level of studies you have achieved? \ud83c\udf93', 'Gross annual salary (in euros) \ud83d\udcb8', \"What's your height? in cm \ud83d\udccf\", \"What's your weight? in Kg \u2696\ufe0f\", 'What is your body complexity? \ud83c\udfcb\ufe0f', 'What is your eye color? \ud83d\udc41\ufe0f', 'What is your hair color? \ud83d\udc69\ud83e\uddb0\ud83d\udc71\ud83c\udffd', 'What is your skin tone?', 'How long is your hair? \ud83d\udc87\ud83c\udffb\u2640\ufe0f\ud83d\udc87\ud83c\udffd\u2642\ufe0f', 'How long is your facial hair? \ud83e\uddd4\ud83c\udffb', 'How often do you wear glasses? \ud83d\udc53', 'How attractive do you consider yourself?', 'Have you ever use an oline dating app?', 'Where have you met your sexual partners? (In a Bar or Restaurant)', 'Where have you met your sexual partners? (Through Friends)', 'Where have you met your sexual partners? (Through Work or as Co-Workers)', 'Where have you met your sexual partners? (Through Family)', 'Where have you met your sexual partners? (in University)', 'Where have you met your sexual partners? (in Primary or Secondary School)', 'Where have you met your sexual partners? (Neighbors)', 'Where have you met your sexual partners? (in Church)', 'Where have you met your sexual partners? (Other)', 'How many people have you kissed?', 'How many sexual partners have you had?', 'How many people have you considered as your boyfriend_girlfriend?', 'How many times per month did you practice sex lately?', 'Happiness scale', 'What area of knowledge is closer to you?', 'If you are in a relationship, how long have you been with your partner?']
Question: What are the top 4 maximum gross annual salaries?
Expected return type: list[number]

Output:
```json
{{
    "solution": "df['Gross annual salary (in euros) \ud83d\udcb8'].nlargest(4).tolist()"
}}
```

Input DataFrame columns: ['segmentation_1', 'descriptor', 'complaint_type', 'created_date', 'borough', 'hour', 'month_name', 'weekday_name', 'agency', 'resolution_description', 'agency_name', 'city', 'location_type', 'incident_zip', 'incident_address', 'street_name', 'cross_street_1', 'cross_street_2', 'intersection_street_1', 'intersection_street_2', 'address_type', 'landmark', 'facility_type', 'status', 'due_date', 'resolution_action_updated_date', 'community_board', 'x_coordinate', 'y_coordinate', 'park_facility_name', 'park_borough', 'bbl', 'open_data_channel_type', 'vehicle_type', 'taxi_company_borough', 'taxi_pickup_location', 'bridge_highway_name', 'bridge_highway_direction', 'road_ramp', 'bridge_highway_segment', 'latitude', 'longitude', 'location', 'unique_key', 'Unnamed_0', 'closed_date']
Question: Which 4 agencies handle the most complaints?
Expected return type: list[category]

Output:
```json
{{
    "solution": "df['agency'].value_counts().nlargest(4).index.tolist()"
}}
```

Input DataFrame columns: ['yr', 'mo', 'dy', 'date', 'st', 'mag', 'inj', 'fat', 'slat', 'slon', 'elat', 'elon', 'len', 'wid']
Question: What is the maximum number of injuries caused by a single tornado?
Expected return type: number

Output:
```json
{{
    "solution": "df['inj'].max()"
}}
```

Input DataFrame columns: ['ID', 'Year_Birth', 'Education', 'Marital_Status', 'Income', 'Kidhome', 'Teenhome', 'Dt_Customer', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Z_CostContact', 'Z_Revenue', 'Response']
Question: Is there any customer with income higher than 100000?
Expected return type: boolean

Output:
```json
{{
    "solution": "df['Income'].gt(100000).any()"
}}
```

Input DataFrame columns: ['neighbourhood_cleansed', 'host_neighbourhood', 'price', 'room_type', 'reviews_per_month', 'property_type', 'bedrooms', 'host_verifications', 'host_acceptance_rate', 'host_identity_verified', 'beds', 'amenities', 'minimum_nights', 'last_review', 'review_scores_rating', 'instant_bookable', 'calculated_host_listings_count', 'first_review', 'number_of_reviews', 'accommodates', 'listing_url', 'last_scraped', 'source', 'name', 'description', 'neighborhood_overview', 'picture_url', 'host_id', 'host_name', 'host_since', 'host_location', 'host_about', 'host_response_time', 'host_response_rate', 'host_is_superhost', 'host_thumbnail_url', 'host_picture_url', 'host_listings_count', 'host_total_listings_count', 'host_has_profile_pic', 'neighbourhood', 'latitude', 'longitude', 'bathrooms', 'bathrooms_text', 'maximum_nights', 'minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights', 'maximum_maximum_nights', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'calendar_updated', 'has_availability', 'availability_30', 'availability_60', 'availability_90', 'availability_365', 'calendar_last_scraped', 'number_of_reviews_ltm', 'number_of_reviews_l30d', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 'license', 'calculated_host_listings_count_entire_homes', 'calculated_host_listings_count_private_rooms', 'calculated_host_listings_count_shared_rooms', 'x', 'y', 'price_M0jI']\nQuestion
Question: Which host verification method is the least used?
Expected return type: category

Output:
```json
{{
    "solution": "df['host_verifications'].str.split(', ').explode().value_counts().idxmin()"
}}
```

Input DataFrame columns: {list_df_columns}
Question: {question}
Expected return type: {row_type}

Output:
"""

prompt_training_1 = """You are an expert data analyst tasked with answering questions based on a dataset. You will be provided a csv file with the data and a question to answer. Your task is to provide the answer to the question.

Dataset CSV file:
```csv
{csv}
```

Question: {question}
Answer: {answer}"""

prompt_eval_1 = """You are an expert data analyst tasked with answering questions based on a dataset. You will be provided a csv file with the data and a question to answer. Your task is to provide the answer to the question.

Dataset CSV file:
```csv
{csv}
```

Question: {question}
Answer: """