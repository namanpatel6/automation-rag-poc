import hashlib

import pandas as pd
import json

# List of file names
file_names = ["./kusto-data/RolloutHealthStatuswithBuildLabel.csv", "./kusto-data/PFServiceBuildLabelPerVE.csv", "./kusto-data/PFCommonDeploymentEvent.csv", "./kusto-data/Ev2RolloutAnalyticsEventStageMap.csv", "./kusto-data/Ev2RolloutAnalyticsEvent.csv", "./kusto-data/AzDeployerRTOAuditEvent.csv", "./kusto-data/PFServiceComponentNames.csv"]

# Iterate through the file names
for file_name in file_names:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_name)

    # Convert the DataFrame to a list of dictionaries
    records = df.to_dict(orient='records')

    # Add unique key to each record
    for record in records:
        # Concatenate values with colon delimiter, handling potential None values
        concatenated_values = ':'.join(str(val) if val is not None else '' for val in record.values())
        # Hash the concatenated string using SHA256
        hashed_values = hashlib.sha256(concatenated_values.encode()).hexdigest()
        # Add the unique key to the record
        record['key'] = hashed_values

    # Convert the list of dictionaries to a JSON string
    json_string = json.dumps(records)

    # Write the JSON string to a file with the same name but .json extension
    output_file_name = file_name.replace(".csv", ".json").replace("./kusto-data/", "./json-data/")
    with open(output_file_name, 'w') as f:
        f.write(json_string)