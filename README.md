# sample-similarity-search
The Property Search system enables AI-driven residential property valuation and search based on real-estate market segmentation and similarity analysis.  The system follows a clean data and model lifecycle leveraging CRDB's vector database capabilities.

![image](https://github.com/user-attachments/assets/02c387f3-9186-4f97-9026-ee01172c854f)


## Market Segmentation
Real-estate data is divided into distinct markets based on the municipality associated with each property’s location.  And each market maintains a focused set of properties to optimize relevance and prediction accuracy.  However, given the spread of property values in large municipalities the data could be further segmented.  Alternatively we could have segmented markets based on building classifications and modeled across locations, but we chose to use municipality so the client app could load interactive maps for each market.

## Raw Data Ingestion
Source property data is loaded into a CockroachDB (CRDB) operational table as raw JSON documents.  Each record is keyed by a unique property ID and indexed by municipality to accelerate market-based queries.

## Change Data Capture (CDC) to Kafka
A CDC changefeed on CRDB publishes changes (new or updated residential properties) to a Kafka topic.  Only residential property data is published to ensure the downstream system focuses on the correct market segment.

## Preprocessing and Vectorization
A Python client subscribes to the Kafka topic and each property’s raw JSON is pre-processed into a scaled feature vector.  Preprocessed vectors are saved to two locations:

1) A CRDB vector database table, with a foreign key referencing the original property JSON.
2) A local pickle file for AI model training.

## Iterative Model Training
A Python service reads the newly generated pickle files and continuously trains a neural network for regression on residential property values.  This enables the model to stay current with the latest market data.

## Real-Time Property Search & Prediction
A client application accepts user input (e.g., search parameters) and performs a similarity search on the CRDB vector database:

1) Finds the top 10 similar property matches within the same municipality.
2) Aggregates the feature vectors of these matches.
3) The averaged vector is passed as input to the AI model.
4) The model returns a predicted property value based on the user's query.



## Running the Simulation
You can download the repository and run the simulation locally to test different scenarios and configurations that are appropriate for your vector / AI workload.  More information on environment setup and the steps required to run the simulation can be found on our [wiki pages](https://github.com/roachlong/sample-similarity-search/wiki)
